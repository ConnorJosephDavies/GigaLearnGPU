#pragma once
#include "../Framework.h"

namespace RLGC {

	// Runs the physics of ALL arenas as one RocketGPU (CUDA) batch instead of RocketSim on CPU threads.
	//
	// The RocketSim Arena/Car/Ball/BoostPad objects stay the source of truth for everything else in
	// GigaLearn (GameState, state setters, rewards, obs, event trackers): they become state *containers*.
	//   - MarkDirty(i): the CPU changed arena i (reset / state setter) -> its state is uploaded next Step().
	//   - Step(ticks):  uploads controls (+ dirty arenas), steps every arena `ticks` ticks on the GPU,
	//                   downloads the results into a host mirror.
	//   - WriteBack(i): copies arena i's downloaded state into its RocketSim objects, advances
	//                   arena->tickCount and fires the arena's car bump/demo callback.
	// RocketSim's own Arena::Step() is never called while this is active.
	class GpuPhysics {
	public:
		explicit GpuPhysics(const std::vector<Arena*>& arenas);
		~GpuPhysics();
		RG_NO_COPY(GpuPhysics);

		void MarkDirty(int arenaIdx);
		void Step(int ticks, bool download = true); // download=false: skip the device->host mirror copy
		void WriteBack(int arenaIdx);

		// Arena i's cars in GPU slot order (= the arena's _cars iteration order, fixed at construction)
		const std::vector<Car*>& Cars(int arenaIdx) const;

		struct Impl;
	private:
		Impl* _impl;
	};
}
