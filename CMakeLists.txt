cmake_minimum_required (VERSION 3.8)

project("GigaLearnGPU")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnExample ${FILES_SRC})

# Set C++ version to 20
set_target_properties(GigaLearnExample PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnExample PROPERTIES CXX_STANDARD 20)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnExample GigaLearnCPP)

