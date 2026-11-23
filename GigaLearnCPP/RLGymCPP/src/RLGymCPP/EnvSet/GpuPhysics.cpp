#include "GpuPhysics.h"

#include <rocketgpu/rocketgpu.h>
#include <rocketgpu/rocketsim_compat.h>
#include <rocketgpu/detail/car_ref.hpp>
#include <rocketgpu/detail/cuda_api.hpp>

#include <cstring>
#include <mutex>

#include <cuda_runtime.h>
#include <new>

namespace rg = rocketgpu;

// gpusim: page-locked (pinned) host buffer, so host<->device copies run at full PCIe speed
template <typename T>
struct PinnedVec {
	T* ptr = nullptr;
	size_t n = 0;
	PinnedVec() = default;
	PinnedVec(const PinnedVec&) = delete;
	PinnedVec& operator=(const PinnedVec&) = delete;
	~PinnedVec() { Release(); }
	void Release() {
		if (!ptr) return;
		for (size_t i = 0; i < n; i++) ptr[i].~T();
		cudaFreeHost(ptr);
		ptr = nullptr; n = 0;
	}
	void resize(size_t count) {
		Release();
		if (!count) return;
		if (cudaMallocHost((void**)&ptr, count * sizeof(T)) != cudaSuccess)
			RG_ERR_CLOSE("GpuPhysics: cudaMallocHost failed (" << count * sizeof(T) << " bytes)");
		n = count;
		for (size_t i = 0; i < n; i++) new (ptr + i) T();
	}
	T* data() { return ptr; }
	size_t size() const { return n; }
	T& operator[](size_t i) { return ptr[i]; }
	const T& operator[](size_t i) const { return ptr[i]; }
};
namespace rgc = rocketgpu::compat;

namespace {
	rg::Vec3 ToRG(const Vec& v) { return rg::Vec3(v.x, v.y, v.z); }
	Vec ToRS(const rg::Vec3& v) { return Vec(v.x, v.y, v.z); }

	rgc::RotMat ToRG(const RotMat& m) {
		rgc::RotMat r; r.forward = ToRG(m.forward); r.right = ToRG(m.right); r.up = ToRG(m.up); return r;
	}
	RotMat ToRS(const rgc::RotMat& m) {
		RotMat r; r.forward = ToRS(m.forward); r.right = ToRS(m.right); r.up = ToRS(m.up); return r;
	}

	// RocketSim hitbox preset -> RocketGPU preset (matched by hitbox size)
	rg::CarConfig PresetFor(const CarConfig& cfg) {
		struct P { const CarConfig* rs; rg::CarConfig(*rgFn)(); };
		const P presets[] = {
			{ &CAR_CONFIG_OCTANE, rg::OctaneConfig }, { &CAR_CONFIG_DOMINUS, rg::DominusConfig },
			{ &CAR_CONFIG_PLANK, rg::PlankConfig },   { &CAR_CONFIG_BREAKOUT, rg::BreakoutConfig },
			{ &CAR_CONFIG_HYBRID, rg::HybridConfig }, { &CAR_CONFIG_MERC, rg::MercConfig },
		};
		for (auto& p : presets)
			if (p.rs->hitboxSize.DistSq(cfg.hitboxSize) < 0.01f)
				return p.rgFn();
		RG_ERR_CLOSE("GpuPhysics: car config does not match a known hitbox preset");
		return rg::OctaneConfig();
	}
}

struct RLGC::GpuPhysics::Impl {
	std::vector<Arena*> arenas;
	std::vector<std::vector<Car*>> cars;      // per arena, slot order
	int maxCars = 0;
	size_t numEnvs = 0;

	std::unique_ptr<rg::BatchSim> sim;
	PinnedVec<rg::CarState> carStates;      // [env * maxCars + slot]
	PinnedVec<rg::BallState> ballStates;    // [env]
	PinnedVec<rg::EnvState> envStates;      // [env]
	PinnedVec<rg::BoostPadState> padStates; // [env * numPads + gpuPad]
	PinnedVec<rg::CarControls> controls;    // [env * maxCars + slot]
	bool mirrorFresh = false;               // host mirror matches the device (last Step downloaded)
	std::vector<int> padRsIdx;                // gpu pad -> RocketSim _boostPads index
	size_t numPads = 0;

	std::vector<uint8_t> dirty;               // per env
	std::vector<uint8_t> bumpSeen;            // [car * 14]
	std::mutex dirtyMutex;

	size_t Slot(size_t env, int slot) const { return env * maxCars + slot; }

	void PullAll() {
		mirrorFresh = true;
		sim->download_car_states(carStates.data(), carStates.size());
		sim->download_ball_states(ballStates.data(), ballStates.size());
		sim->download_env_states(envStates.data(), envStates.size());
		if (numPads)
			sim->download_boost_pad_states(padStates.data(), padStates.size());
	}

	// RocketSim objects of env -> host mirror (mirrors compat Car/Ball/BoostPad::SetState)
	void ImportEnv(size_t env) {
		Arena* arena = arenas[env];
		for (int slot = 0; slot < (int)cars[env].size(); slot++) {
			Car* car = cars[env][slot];
			CarState in = car->GetState();
			rg::CarState& s = carStates[Slot(env, slot)];
			s.pad_frame_pos_valid = 0;
			s.pos = ToRG(in.pos);
			rgc::SeedCarBasisFromRotMat(s, ToRG(in.rotMat));
			s.set_wheel_normals_mask = 0;
			s.vel = ToRG(in.vel);
			s.ang_vel = ToRG(in.angVel);
			s.reported_flags = 0;
			s.is_on_ground = in.isOnGround;
			for (int w = 0; w < 4; w++) s.wheels_with_contact[w] = in.wheelsWithContact[w];
			s.has_jumped = in.hasJumped; s.has_double_jumped = in.hasDoubleJumped; s.has_flipped = in.hasFlipped;
			s.flip_rel_torque = ToRG(in.flipRelTorque);
			s.jump_hold_broken = 0;
			s.jump_time = in.jumpTime; s.flip_time = in.flipTime; s.flip_timer_live = 0;
			s.is_flipping = in.isFlipping; s.is_jumping = in.isJumping;
			s.air_time = in.airTime; s.air_time_since_jump = in.airTimeSinceJump;
			s.boost = in.boost;
			s.time_since_boosted = 0; s.is_boosting = 0; s.boosting_time = in.timeSpentBoosting;
			s.is_supersonic = in.isSupersonic; s.supersonic_time = in.supersonicTime;
			s.handbrake_val = in.handbrakeVal;
			s.is_auto_flipping = in.isAutoFlipping; s.auto_flip_timer = in.autoFlipTimer;
			s.auto_flip_torque_scale = in.autoFlipTorqueScale;
			s.world_contact.has_contact = in.worldContact.hasContact;
			s.world_contact.contact_normal = ToRG(in.worldContact.contactNormal);
			s.car_contact.other_car_id = 0; s.car_contact.cooldown_timer = in.carContact.cooldownTimer;
			s.is_demoed = in.isDemoed; s.demo_respawn_timer = in.demoRespawnTimer;
			s.ball_hit_info.is_valid = in.ballHitInfo.isValid;
			s.ball_hit_info.relative_pos_on_ball = ToRG(in.ballHitInfo.relativePosOnBall);
			s.ball_hit_info.ball_pos = ToRG(in.ballHitInfo.ballPos);
			s.ball_hit_info.extra_hit_vel = ToRG(in.ballHitInfo.extraHitVel);
			s.ball_hit_info.tick_count_when_hit = in.ballHitInfo.tickCountWhenHit;
			s.ball_hit_info.tick_count_when_extra_impulse_applied = in.ballHitInfo.tickCountWhenExtraImpulseApplied;
			s.ball_hit_info.extra_tick_is_previous = 0;
			s.tick_count_since_update = 0;
			s.pending_bump_count = 0;
			s.vel_impulse_cache = rg::Vec3(0, 0, 0);
			s.vel_impulse_cache_bt = rg::Vec3(0, 0, 0);
			std::memcpy(&bumpSeen[Slot(env, slot) * 14], s.bump_event_bumps, 7);
			std::memcpy(&bumpSeen[Slot(env, slot) * 14 + 7], s.bump_event_demos, 7);
		}

		BallState inBall = arena->ball->GetState();
		rg::BallState& b = ballStates[env];
		b.pos = ToRG(inBall.pos);
		b.basis_forward = ToRG(inBall.rotMat.forward);
		b.basis_right = ToRG(inBall.rotMat.right);
		b.basis_up = ToRG(inBall.rotMat.up);
		b.basis_authoritative = 1;
		b.rot = rgc::QuatFromRotMat(ToRG(inBall.rotMat));
		b.vel = ToRG(inBall.vel);
		b.ang_vel = ToRG(inBall.angVel);
		b.vel_impulse_cache = rg::Vec3(0, 0, 0);
		b.vel_impulse_cache_bt = rg::Vec3(0, 0, 0);
		b.sleeping = 2; // "just uploaded": the next step infers the activation state
		b.deactivation_time = 0;
		b.tick_count_since_update = 0;

		for (size_t p = 0; p < numPads; p++) {
			auto in = arena->_boostPads[padRsIdx[p]]->GetState();
			rg::BoostPadState& ps = padStates[env * numPads + p];
			ps.is_active = in.isActive;
			ps.cooldown = in.cooldown;
			ps.prev_locked_car_id = 0;
			ps.cur_locked_car_id = 0;
		}
	}
};

RLGC::GpuPhysics::GpuPhysics(const std::vector<Arena*>& arenas) : _impl(new Impl()) {
	Impl& m = *_impl;
	m.arenas = arenas;
	m.numEnvs = arenas.size();
	m.cars.resize(m.numEnvs);
	for (size_t i = 0; i < m.numEnvs; i++) {
		for (Car* car : arenas[i]->_cars)
			m.cars[i].push_back(car);
		m.maxCars = RS_MAX(m.maxCars, (int)m.cars[i].size());
		RG_ASSERT(arenas[i]->gameMode == GameMode::SOCCAR);
	}
	RG_ASSERT(m.maxCars >= 1 && m.maxCars <= 8);

	rg::SimConfig cfg;
	cfg.env_count = (int32_t)m.numEnvs;
	cfg.max_cars_per_env = m.maxCars;
	cfg.mode = rg::Mode::Soccar;
	m.sim = std::make_unique<rg::BatchSim>(cfg);

	const size_t cap = m.sim->car_capacity();
	RG_ASSERT(cap == m.numEnvs * m.maxCars);
	std::vector<int32_t> counts(m.numEnvs);
	std::vector<rg::Team> teams(cap, rg::Team::Blue);
	std::vector<rg::CarConfig> configs(cap, rg::OctaneConfig());
	for (size_t e = 0; e < m.numEnvs; e++) {
		counts[e] = (int32_t)m.cars[e].size();
		for (int s = 0; s < counts[e]; s++) {
			Car* car = m.cars[e][s];
			teams[m.Slot(e, s)] = car->team == Team::BLUE ? rg::Team::Blue : rg::Team::Orange;
			configs[m.Slot(e, s)] = PresetFor(car->config);
		}
	}
	m.sim->set_env_car_count(counts.data(), counts.size());
	m.sim->set_teams(teams.data(), teams.size());
	m.sim->set_car_configs(configs.data(), configs.size());

	// Boost pads: map RocketGPU's pad order onto RocketSim's by position
	const rg::SimConfig& simCfg = m.sim->config();
	m.numPads = m.sim->boost_pad_capacity() / m.numEnvs;
	RG_ASSERT(m.numPads == arenas[0]->_boostPads.size());
	m.padRsIdx.assign(m.numPads, -1);
	for (size_t p = 0; p < m.numPads; p++) {
		Vec gpuPos = ToRS(simCfg.boost_pads[p].pos);
		for (size_t r = 0; r < arenas[0]->_boostPads.size(); r++)
			if (arenas[0]->_boostPads[r]->config.pos.DistSq2D(gpuPos) < 10)
				m.padRsIdx[p] = (int)r;
		RG_ASSERT(m.padRsIdx[p] >= 0);
	}

	m.carStates.resize(cap);
	m.ballStates.resize(m.numEnvs);
	m.envStates.resize(m.numEnvs);
	m.padStates.resize(m.numEnvs * m.numPads);
	m.controls.resize(cap);
	m.bumpSeen.assign(cap * 14, 0);
	m.dirty.assign(m.numEnvs, 1);

	// Same tick clock as RocketSim (ball hit ticks are compared against arena->tickCount)
	m.PullAll();
	for (size_t e = 0; e < m.numEnvs; e++)
		RG_ASSERT(m.envStates[e].tick == arenas[e]->tickCount);
}

RLGC::GpuPhysics::~GpuPhysics() { delete _impl; }

const std::vector<Car*>& RLGC::GpuPhysics::Cars(int arenaIdx) const { return _impl->cars[arenaIdx]; }

void RLGC::GpuPhysics::MarkDirty(int arenaIdx) {
	_impl->dirty[arenaIdx] = 1; // distinct bytes per arena: safe from parallel jobs
}

void RLGC::GpuPhysics::Step(int ticks, bool download) {
	Impl& m = *_impl;
	if (ticks <= 0)
		return;

	// 1. Upload CPU-side changes (resets / state setters)
	bool anyDirty = false;
	for (uint8_t d : m.dirty) anyDirty |= (d != 0);
	if (anyDirty) {
		if (!m.mirrorFresh)
			m.PullAll(); // the full PreserveInternal upload below must start from the current device state
		// The mirror holds every env's last downloaded state (incl. internal fields), so a full
		// PreserveInternal upload leaves clean envs exactly as they were.
		for (size_t e = 0; e < m.numEnvs; e++)
			if (m.dirty[e])
				m.ImportEnv(e);
		m.sim->upload_car_states(m.carStates.data(), m.carStates.size(), nullptr,
			rg::BatchSim::StateUploadMode::PreserveInternal);
		if (m.numPads)
			m.sim->upload_boost_pad_states(m.padStates.data(), m.padStates.size());
		auto span = rg::detail::CarSpanFromBytes(m.sim->device_view().car_states);
		for (size_t e = 0; e < m.numEnvs; e++) {
			if (!m.dirty[e]) continue;
			for (int s = 0; s < (int)m.cars[e].size(); s++)
				rg::detail::LaunchCanonicalizeImportedCars(span + (int64_t)m.Slot(e, s), 1, nullptr);
			m.sim->upload_ball_state_public_at(m.ballStates[e], e); // only this env's ball
			m.dirty[e] = 0;
		}
	}

	// 2. Controls
	for (size_t e = 0; e < m.numEnvs; e++) {
		for (int s = 0; s < (int)m.cars[e].size(); s++) {
			const CarControls& c = m.cars[e][s]->controls;
			rg::CarControls& k = m.controls[m.Slot(e, s)];
			k.throttle = c.throttle; k.steer = c.steer; k.pitch = c.pitch; k.yaw = c.yaw; k.roll = c.roll;
			k.jump = c.jump; k.boost = c.boost; k.handbrake = c.handbrake;
		}
	}
	m.sim->upload_controls(m.controls.data(), m.controls.size());

	// 3. Step every arena at once, then mirror back
	m.sim->step(ticks);
	if (download)
		m.PullAll();
	else
		m.mirrorFresh = false; // nothing reads the mirror until the next downloading Step
}

void RLGC::GpuPhysics::WriteBack(int arenaIdx) {
	Impl& m = *_impl;
	size_t env = arenaIdx;
	Arena* arena = m.arenas[env];
	auto& cars = m.cars[env];

	for (int slot = 0; slot < (int)cars.size(); slot++) {
		const rg::CarState& s = m.carStates[m.Slot(env, slot)];
		Car* car = cars[slot];
		CarState out = car->GetState();
		out.pos = ToRS(s.pos);
		out.rotMat = ToRS(rgc::PublicRotMatFromCarState(s));
		out.vel = ToRS(s.vel);
		out.angVel = ToRS(s.ang_vel);
		out.isOnGround = s.is_on_ground;
		for (int w = 0; w < 4; w++) out.wheelsWithContact[w] = s.wheels_with_contact[w];
		out.hasJumped = s.has_jumped; out.hasDoubleJumped = s.has_double_jumped; out.hasFlipped = s.has_flipped;
		out.flipRelTorque = ToRS(s.flip_rel_torque);
		out.jumpTime = s.jump_time; out.flipTime = s.flip_time;
		out.isFlipping = s.is_flipping; out.isJumping = s.is_jumping;
		out.airTime = s.air_time; out.airTimeSinceJump = s.air_time_since_jump;
		out.boost = s.boost;
		out.timeSpentBoosting = s.boosting_time;
		out.isSupersonic = s.is_supersonic; out.supersonicTime = s.supersonic_time;
		out.handbrakeVal = s.handbrake_val;
		out.isAutoFlipping = s.is_auto_flipping; out.autoFlipTimer = s.auto_flip_timer;
		out.autoFlipTorqueScale = s.auto_flip_torque_scale;
		out.worldContact.hasContact = s.world_contact.has_contact;
		out.worldContact.contactNormal = ToRS(s.world_contact.contact_normal);
		{
			uint32_t dev = s.car_contact.other_car_id; // 1 + slot within env, 0 = none
			out.carContact.otherCarID = (dev >= 1 && dev <= cars.size()) ? cars[dev - 1]->id : 0;
			out.carContact.cooldownTimer = s.car_contact.cooldown_timer;
		}
		out.isDemoed = s.is_demoed; out.demoRespawnTimer = s.demo_respawn_timer;
		out.ballHitInfo.isValid = s.ball_hit_info.is_valid;
		out.ballHitInfo.relativePosOnBall = ToRS(s.ball_hit_info.relative_pos_on_ball);
		out.ballHitInfo.ballPos = ToRS(s.ball_hit_info.ball_pos);
		out.ballHitInfo.extraHitVel = ToRS(s.ball_hit_info.extra_hit_vel);
		out.ballHitInfo.tickCountWhenHit = s.ball_hit_info.tick_count_when_hit;
		out.ballHitInfo.tickCountWhenExtraImpulseApplied = s.ball_hit_info.tick_count_when_extra_impulse_applied;
		out.lastControls = car->controls;
		car->SetState(out);
	}

	{
		const rg::BallState& b = m.ballStates[env];
		BallState out = arena->ball->GetState();
		out.pos = ToRS(b.pos);
		out.rotMat.forward = ToRS(b.basis_forward);
		out.rotMat.right = ToRS(b.basis_right);
		out.rotMat.up = ToRS(b.basis_up);
		out.vel = ToRS(b.vel);
		out.angVel = ToRS(b.ang_vel);
		arena->ball->SetState(out);
	}

	for (size_t p = 0; p < m.numPads; p++) {
		const rg::BoostPadState& ps = m.padStates[env * m.numPads + p];
		BoostPad* pad = arena->_boostPads[m.padRsIdx[p]];
		BoostPadState out = pad->GetState();
		out.isActive = ps.is_active;
		out.cooldown = ps.cooldown;
		out.curLockedCar = NULL;
		pad->SetState(out);
	}

	arena->tickCount = m.envStates[env].tick;

	// Bump / demo events (same decoding as RocketGPU's compat Arena::DrainBumpEvents)
	auto& cb = arena->_carBumpCallback;
	for (int slot = 0; slot < (int)cars.size(); slot++) {
		const rg::CarState& s = m.carStates[m.Slot(env, slot)];
		uint8_t now[14];
		std::memcpy(now, s.bump_event_bumps, 7);
		std::memcpy(now + 7, s.bump_event_demos, 7);
		uint8_t* seen = &m.bumpSeen[m.Slot(env, slot) * 14];
		for (int i = 0; i < 14; i++) {
			uint8_t fresh = (uint8_t)(now[i] - seen[i]);
			seen[i] = now[i];
			if (!fresh || !cb.func) continue;
			int victimSlot = (slot + (i % 7) + 1) & 7;
			if (victimSlot >= (int)cars.size()) continue;
			for (int n = 0; n < fresh; n++)
				cb.func(arena, cars[slot], cars[victimSlot], i >= 7, cb.userInfo);
		}
	}
}
