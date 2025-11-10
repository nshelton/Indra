cmake_minimum_required(VERSION 3.21)
project(Indra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
  src/*.cpp
)

add_executable(indra ${SRC_FILES})

target_include_directories(indra PRIVATE src)

target_link_libraries(indra PRIVATE glfw imgui::imgui opengl32 glog::glog fmt::fmt glad::glad)

enable_testing()

add_executable(indra_tests
  tests/test_matrix4.cpp
)

target_include_directories(indra_tests PRIVATE src)
target_link_libraries(indra_tests PRIVATE GTest::gtest GTest::gtest_main)

add_test(NAME indra_matrix4 COMMAND indra_tests)