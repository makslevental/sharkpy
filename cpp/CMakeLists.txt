cmake_minimum_required(VERSION 3.20)
project(binding)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Python method:
pybind11_add_module(sharkpy_ext sharkpy.cpp)
#target_include_directories(sharkpy PRIVATE /Users/mlevental/dev_projects/llvm-project/llvm-install/include)

install(TARGETS sharkpy_ext DESTINATION ".")
