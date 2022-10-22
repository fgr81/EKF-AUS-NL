cmake_minimum_required(VERSION 3.5.0)

project(ekf-aus-nl)

set (CMAKE_CXX_STANDARD 14)

#find_package(PythonLibs)
#include_directories(${PYTHON_INCLUDE_DIRS})

find_package( PkgConfig)
#include_directories( Eigen )


find_package(pybind11)
find_package(Eigen3 REQUIRED) 
include_directories(${EIGEN3_INCLUDE_DIR})
pybind11_add_module(ekfaus wrap1.cpp)
set(CMAKE_CXX_FLAGS "-g -O0")

