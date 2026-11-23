# GigaLearnGPU

GigaLearnGPU is [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak) (C++ PPO for Rocket League bots) with an optional GPU physics backend. When the backend is on, every arena is stepped in one CUDA batch by [RocketGPU](https://github.com/thecosmicskye/RocketGPU) instead of one RocketSim instance per arena on the CPU. That lets you run thousands of arenas at once and gets much more SPS out of the same machine.

- **On** (`GIGALEARN_GPU_PHYSICS=1`): arenas are batched on the GPU. 4096 arenas is the default.
- **Off**: the original CPU RocketSim path runs unchanged.

## Performance

I measured this on one RTX 3070 with a Ryzen 7 2700X, running 1v1 at 4096 arenas with tick skip 8:

- **About 95–105k steps per second (SPS) end to end.** That number includes PPO training (2 epochs, 400k batch) and a 1024-wide network.
- **The CPU RocketSim path was a small fraction of that** on the same machine.

## What's changed from GigaLearnCPP

- **GPU physics bridge.** Lives in `RLGymCPP/src/RLGymCPP/EnvSet/GpuPhysics.{h,cpp}`, with `EnvSet` changes so that state setters, rewards and observations still see normal RocketSim `Arena`/`Car` objects. Dirty arenas are uploaded and stepped results are written back.
- **Faster rollouts.**
  - Half-precision rollout inference (`GIGALEARN_HALF_INFER=1`, default on). Training stays in fp32.
  - Optional async PPO (`GIGALEARN_ASYNC_PPO`).
- **MSVC/Windows patches to RocketGPU.** They're vendored in `RocketGPU/` and listed in [PATCHES.md](PATCHES.md).

## Building (Windows, MSVC + Ninja)

You need:
- the CUDA toolkit (12.6+; 12.9 tested);
- [libtorch](https://pytorch.org/) (CUDA build), unzipped into `GigaLearnCPP/libtorch`;
- Python 3 for metrics, optional.

```bat
git clone --recursive https://github.com/<you>/GigaLearnGPU
cd GigaLearnGPU

:: 1. Build RocketGPU (produces RocketGPU/build/rocketgpu.lib)
cmake -S RocketGPU -B RocketGPU/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DROCKETGPU_USE_CUDA=ON -DROCKETGPU_CPU_ISPC=OFF -DROCKETGPU_CUDA_SPLIT=ON -DCMAKE_CUDA_ARCHITECTURES=86
:: (86 = RTX 30xx; use 89 for RTX 40xx, 120 for RTX 50xx)
cmake --build RocketGPU/build

:: 2. Build GigaLearn + the example
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

To build against a different CUDA install, set `ROCKETGPU_CUDA_ROOT` (it defaults to `%CUDA_PATH%`).

## Running

RocketSim needs collision meshes dumped from your own copy of Rocket League (see the [RocketSim docs](https://github.com/ZealanL/RocketSim)). Point the example at them with `ROCKETSIM_MESHES`, then run:

```bat
set ROCKETSIM_MESHES=C:\path\to\collision_meshes
set GIGALEARN_GPU_PHYSICS=1
build\GigaLearnExample.exe
```

| Env var | Default | Meaning |
|---|---|---|
| `GIGALEARN_GPU_PHYSICS` | off | `1` = RocketGPU batched physics |
| `GIGALEARN_NUM_GAMES` | 4096 (GPU) / 256 (CPU) | arenas |
| `GIGALEARN_TS_PER_ITR` | 400k (GPU) / 50k (CPU) | steps collected per PPO iteration |
| `GIGALEARN_HALF_INFER` | 1 | fp16 rollout inference |
| `GIGALEARN_ASYNC_PPO` | off | overlap collection and training |

`src/ExampleMain.cpp` is a minimal 1v1 trainer that uses only the stock rewards. Swap in your own rewards, obs and state setters exactly as you would in GigaLearnCPP.

## Credits and licenses

- **GigaLearnCPP and RLGymCPP:** ZealanL.
- **RocketSim:** ZealanL, MIT.
- **RocketGPU:** Cosmic Skye, MIT (see `RocketGPU/LICENSE` and `RocketGPU/NOTICE`).
- **Bullet, nlohmann/json, thread-pool and pybind11:** under their own licenses in their folders.

This is an unofficial derivative and is not affiliated with Psyonix or Epic Games.
