# Changes to upstream code

## RocketGPU (v1.3.0) — Windows/MSVC patches
1. `include/rocketgpu/detail/step/env_step.hpp` (`CopyCarWorldRow`) and `arena_geometry.hpp` (BulletSweepHistory copy):
   `__builtin_assume_aligned(&typed, 16)` → cast to `void*` first. NVCC's MSVC-mode front end otherwise emits a typed
   store into the builtin's `void*` temporary and NVVM aborts ("Stored value type does not match pointer operand type")
   on CUDA 12.6 and 12.9. Same memcpy, same alignment hint.
2. `tools/bench_cpp.cpp` `MulHi`: `__umulh` on MSVC (no `unsigned __int128`).
3. New `BatchSim::upload_ball_state_public_at(state, env)`: canonicalise/prime ONE env's ball (the stock
   `upload_ball_states(PublicState)` re-primes every env's ball contacts).
4. `gpusim_cmp/` + CMake target `rg_compare`: RocketGPU vs RocketSim side by side (not part of the library).

## GigaLearnCPP / RLGymCPP
- `RLGymCPP/src/RLGymCPP/EnvSet/GpuPhysics.{h,cpp}`: owns a RocketGPU `BatchSim` for all arenas. Arenas touched by
  state setters are marked dirty and uploaded; after each batched step, car/ball state is written back into the
  RocketSim objects so rewards, obs and terminal conditions work unchanged. Enabled by `GIGALEARN_GPU_PHYSICS=1`.
- `EnvSet`: steps through `GpuPhysics` when enabled, otherwise the original per-arena CPU path.
- `PPOLearner`: fp16 rollout inference (`GIGALEARN_HALF_INFER`), optional async PPO (`GIGALEARN_ASYNC_PPO`),
  optional guiding-policy (distillation) settings in `PPOLearnerConfig`.
