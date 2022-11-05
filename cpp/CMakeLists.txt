cmake_minimum_required(VERSION 3.20)
project(binding)

set(CMAKE_CXX_STANDARD 17)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Python3_SITELIB ${Python3_SITELIB}")

if (NOT DEFINED LLVM_INCLUDE_DIR)
    message(FATAL_ERROR "missing LLVM_INCLUDE_DIR")
else ()
    message(STATUS "LLVM_INCLUDE_DIR - ${LLVM_INCLUDE_DIR}")
endif ()

if (NOT DEFINED LLVM_LIB_DIR)
    message(FATAL_ERROR "missing LLVM_LIB_DIR")
else ()
    message(STATUS "LLVM_LIB_DIR - ${LLVM_LIB_DIR}")
endif ()

pybind11_add_module(sharkpy_ext sharkpy_ext.cpp)
target_include_directories(sharkpy_ext PRIVATE ${LLVM_INCLUDE_DIR})
target_link_directories(sharkpy_ext PRIVATE ${LLVM_LIB_DIR})

#file(STRINGS libs.txt NEEDED_LIBS NEWLINE_CONSUME)
#STRING(REGEX REPLACE "\n" ";" NEEDED_LIBS "${NEEDED_LIBS}")
#target_link_libraries(sharkpy_ext PRIVATE ${NEEDED_LIBS})
#target_link_libraries(sharkpy_ext PRIVATE tinfo)

message(STATUS "CMAKE_C_COMPILER_ABI ${CMAKE_C_COMPILER_ABI}")
message(STATUS "CMAKE_CXX_COMPILER_ABI ${CMAKE_CXX_COMPILER_ABI}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif ()

install(TARGETS sharkpy_ext DESTINATION "${Python3_SITELIB}")
