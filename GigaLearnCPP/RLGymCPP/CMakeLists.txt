cmake_minimum_required (VERSION 3.8)

project("RLGymCPP")

# Add all headers and code files
file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h")
add_library(RLGymCPP STATIC ${FILES_SRC})
target_include_directories(RLGymCPP PUBLIC "src/")

# Set C++ version to 20
set_target_properties(RLGymCPP PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(RLGymCPP PROPERTIES CXX_STANDARD 20)

# Include RocketSim
add_subdirectory("RocketSim")
target_link_libraries(RLGymCPP RocketSim)

# Include thread pool library (https://github.com/DeveloperPaul123/thread-pool)
target_include_directories(RLGymCPP PUBLIC "thread_pool")
# gpusim: RocketGPU (CUDA) batched physics for EnvSet (see src/RLGymCPP/EnvSet/GpuPhysics.*)
set(ROCKETGPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../RocketGPU" CACHE PATH "RocketGPU source (built in ROCKETGPU_ROOT/build)")
set(ROCKETGPU_CUDA_ROOT "$ENV{CUDA_PATH}" CACHE PATH "CUDA toolkit with cudart headers/libs")
target_include_directories(RLGymCPP PRIVATE "${ROCKETGPU_ROOT}/include" "${ROCKETGPU_CUDA_ROOT}/include")
# Must match how rocketgpu.lib was built: the CUDA build changes the state structs' layout
target_compile_definitions(RLGymCPP PRIVATE ROCKETGPU_USE_CUDA=1)
target_link_libraries(RLGymCPP "${ROCKETGPU_ROOT}/build/rocketgpu.lib" "${ROCKETGPU_CUDA_ROOT}/lib/x64/cudart.lib")
