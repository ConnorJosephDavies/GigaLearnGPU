#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"
#include "../TerminalConditions/TerminalCondition.h"
#include "../Rewards/Reward.h"
#include "../OBSBuilders/OBSBuilder.h"
#include "../ActionParsers/ActionParser.h"
#include "../StateSetters/StateSetter.h"
#include "../ThreadPool.h"
#include "GpuPhysics.h"
#include <RLGymCPP/Rewards/Reward.h>

namespace RLGC {

	struct EnvCreateResult {
		Arena* arena;
		std::vector<WeightedReward> rewards;
		std::vector<TerminalCondition*> terminalConditions;
		ObsBuilder* obsBuilder;
		ActionParser* actionParser;
		StateSetter* stateSetter;

		void* userInfo; // Optional userinfo pointer if you want to track some data per-env
	};
	typedef std::function<EnvCreateResult(int index)> EnvCreateFn;

	struct EnvSetConfig {
		EnvCreateFn envCreateFn;
		int numArenas;
		int tickSkip;
		int actionDelay;
		bool saveRewards;
		bool shuffleRewardSampling = true;
		bool savePlayerRewards = false; // Watch mode: keep every player's WEIGHTED per-reward value for arena 0 (visualiser)
		bool useGpuPhysics = false; // Step all arenas on RocketGPU (CUDA). Env var GIGALEARN_GPU_PHYSICS=1 also enables it.
	};

	struct EnvState {
		int numPlayers;
		std::vector<GameState> gameStates;
		std::vector<GameState> prevGameStates;
		DimList2<float> obs;
		DimList2<uint8_t> actionMasks;
		std::vector<float> rewards;
		std::vector<std::vector<float>> lastRewards; // Only from the first arena
		std::vector<std::string> playerRewardNames; // savePlayerRewards: reward names for arena 0
		std::vector<std::vector<float>> playerRewards; // savePlayerRewards: [player][reward] weighted value, arena 0
		std::vector<uint8_t> terminals;

		std::vector<int> arenaPlayerStartIdx = {};

		void Resize(std::vector<Arena*>& arenas) {
			numPlayers = 0;
			for (int i = 0; i < arenas.size(); i++) {
				arenaPlayerStartIdx.push_back(numPlayers);
				numPlayers += arenas[i]->_cars.size();
			}

			gameStates.resize(arenas.size());
			prevGameStates.resize(arenas.size());
			rewards.resize(numPlayers);
			lastRewards.resize(arenas.size());
			terminals.resize(arenas.size());
		}
	};

	struct EnvSet {

		struct CallbackUserInfo {
			RLGC::EnvSet* envSet;
			Arena* arena;
			int arenaIdx;
		};

		std::vector<Arena*> arenas;
		std::vector<GameEventTracker*> eventTrackers;

		std::vector<CallbackUserInfo*> eventCallbackInfos;
		std::vector<void*> userInfos;

		EnvSetConfig config;

		int obsSize;
		int numActions;

		std::vector<std::vector<WeightedReward>> rewards;
		std::vector<std::vector<TerminalCondition*>> terminalConditions;
		std::vector<ObsBuilder*> obsBuilders;
		std::vector<ActionParser*> actionParsers;
		std::vector<StateSetter*> stateSetters;

		EnvState state = {};

		GpuPhysics* gpu = NULL; // Non-null when arena physics runs on the GPU
		std::vector<std::vector<Action>> pendingActions;

		EnvSet(const EnvSetConfig& config);

		RG_NO_COPY(EnvSet);

		~EnvSet() {
			delete gpu;
			for (Arena* arena : arenas)
				delete arena;

			for (auto& eventTracker : eventTrackers)
				delete eventTracker;
			for (auto& eventCallbackInfo : eventCallbackInfos)
				delete eventCallbackInfo;
		}

		////////////////////
		
		void StepFirstHalf(bool async);
		void StepSecondHalf(const IList& actionIndices, bool async);
		void Sync() { g_ThreadPool.WaitUntilDone(); }
		void ResetArena(int index);
		void PostStepArena(int arenaIdx, const std::vector<Action>& actions);
		void Reset();
	};
}