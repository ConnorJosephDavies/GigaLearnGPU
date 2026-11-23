// Minimal example trainer for GigaLearnGPU.
// Set GIGALEARN_GPU_PHYSICS=1 to step every arena on the GPU with RocketGPU (thousands of arenas, much higher SPS).
// Without it, the original CPU RocketSim path runs unchanged.
#include <cstdlib>
#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/OBSBuilders/AdvancedObs.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

using namespace GGL;
using namespace RLGC;

static int EnvInt(const char* name, int fallback) {
	const char* v = std::getenv(name);
	return v ? std::atoi(v) : fallback;
}

EnvCreateResult EnvCreateFunc(int index) {
	std::vector<WeightedReward> rewards = {
		{ new AirReward(), 0.25f },
		{ new FaceBallReward(), 0.25f },
		{ new VelocityPlayerToBallReward(), 4.f },
		{ new StrongTouchReward(20, 100), 60 },
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 2.0f },
		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },
		{ new ZeroSumReward(new GoalReward(), 1), 150 },
	};

	std::vector<TerminalCondition*> terminalConditions = {
		new NoTouchCondition(10),
		new GoalScoreCondition()
	};

	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	EnvCreateResult result = {};
	result.actionParser = new DefaultAction();
	result.obsBuilder = new AdvancedObs();
	result.stateSetter = new KickoffState();
	result.terminalConditions = terminalConditions;
	result.rewards = rewards;
	result.arena = arena;
	return result;
}

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {}

int main(int argc, char* argv[]) {
	// RocketSim needs the collision meshes dumped from your own copy of Rocket League (see README).
	const char* meshes = std::getenv("ROCKETSIM_MESHES");
	RocketSim::Init(meshes ? meshes : "collision_meshes");

	const char* gpu = std::getenv("GIGALEARN_GPU_PHYSICS");
	bool useGpu = gpu && gpu[0] == '1';

	LearnerConfig cfg = {};
	cfg.deviceType = LearnerDeviceType::GPU_CUDA;
	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1;

	// GPU physics batches every arena into one kernel launch, so it wants far more arenas than the CPU path.
	cfg.numGames = EnvInt("GIGALEARN_NUM_GAMES", useGpu ? 4096 : 256);
	int tsPerItr = EnvInt("GIGALEARN_TS_PER_ITR", useGpu ? 400'000 : 50'000);
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = 50'000; // lower this if you run out of VRAM
	cfg.ppo.epochs = 2;
	cfg.ppo.entropyScale = 0.035f;
	cfg.ppo.gaeGamma = 0.99;
	cfg.ppo.policyLR = 1.5e-4;
	cfg.ppo.criticLR = 1.5e-4;

	cfg.ppo.sharedHead.layerSizes = { 256, 256 };
	cfg.ppo.policy.layerSizes = { 256, 256, 256 };
	cfg.ppo.critic.layerSizes = { 256, 256, 256 };

	cfg.randomSeed = 123;
	cfg.checkpointFolder = "checkpoints";
	cfg.sendMetrics = false; // set true (and install wandb) to log metrics

	if (argc > 1 && std::string(argv[1]) == "render") {
		cfg.renderMode = true;
		cfg.renderTimeScale = 1.0f;
		cfg.numGames = 1;
	}

	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);
	learner->Start();
	return 0;
}
