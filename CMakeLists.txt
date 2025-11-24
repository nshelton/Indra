cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER "$ENV{CUDA_PATH}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
message(STATUS "Using CUDA from CUDA_PATH: $ENV{CUDA_PATH}")
message(STATUS "Setting CUDA compiler from CUDA_PATH: ${CMAKE_CUDA_COMPILER}")

project(Indra LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Verify CUDA was found and print information
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found!")
    message(STATUS "  Version: ${CUDAToolkit_VERSION}")
    message(STATUS "  Include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "  Library dirs: ${CUDAToolkit_LIBRARY_DIR}")
else()
    message(FATAL_ERROR "CUDA Toolkit not found! Please set CUDA_TOOLKIT_ROOT_DIR or CUDA_PATH environment variable.")
endif()

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  src/*.cpp
  src/*.cu
)

add_executable(indra ${SRC_FILES})

# Prevent Windows.h from defining min/max macros that conflict with std::min/std::max
target_compile_definitions(indra PRIVATE NOMINMAX)

target_include_directories(indra PRIVATE
  src
  ${CUDAToolkit_INCLUDE_DIRS}
)

target_precompile_headers(indra PRIVATE
  <glad/glad.h>
  <glog/logging.h>
  <nlohmann/json.hpp>
  <imgui.h>
  <memory>
  <vector>
  <string>
)

target_link_libraries(indra PRIVATE
  glfw
  imgui::imgui
  opengl32
  glog::glog
  fmt::fmt
  glad::glad
  CUDA::cudart
  CUDA::cuda_driver
  CUDA::cufft
  CUDA::nvrtc
  nlohmann_json::nlohmann_json
)

# Copy shaders directory to build output for hot-reload support
add_custom_command(TARGET indra POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders
    $<TARGET_FILE_DIR:indra>/shaders
  COMMENT "Copying shaders directory to build output"
)

enable_testing()

add_executable(indra_tests
  tests/test_matrix4.cpp
)

target_include_directories(indra_tests PRIVATE src)
target_link_libraries(indra_tests PRIVATE GTest::gtest GTest::gtest_main)

add_test(NAME indra_matrix4 COMMAND indra_tests)