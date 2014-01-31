cmake_minimum_required(VERSION 2.8)
project(GPU-Multigrid-GVF)

add_subdirectory(SIPL)
include_directories(${SIPL_INCLUDE_DIRS})

add_subdirectory(OpenCLUtilityLibrary)
find_package(OpenCLUtilityLibrary PATHS "${GPU-Multigrid-GVF_BINARY_DIR}/OpenCLUtilityLibrary" REQUIRED)
include(${OpenCLUtilityLibrary_USE_FILE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++0x -O3 -DKERNELS_DIR=\\\"${PROJECT_SOURCE_DIR}/\\\"")

add_executable(GVF main.cpp gradientVectorFlow.cpp)
target_link_libraries(GVF SIPL OpenCLUtilityLibrary)
