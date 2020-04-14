cmake_minimum_required(VERSION 3.13.4)
project(gamma VERSION 0.1 LANGUAGES CXX CUDA)

# CMake Settings
set(CMAKE_SUPPRESS_REGENERATION True)

# Installation Configurations
set(GAMMA_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

# Set paths for external dependencies
set(STB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/stb)
set(TINYGLTF_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/tinygltf)

set(TINYGLTF_USE_CPP14 ON)

# Include external dependencies
include_directories(${STB_INCLUDE_DIR})
include_directories(SYSTEM ${TINYGLTF_INCLUDE_DIR}) # Marking it as SYSTEM supress all warnings from that include
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)