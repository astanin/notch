cmake_minimum_required(VERSION 3.0)
project(notch)

IF(CMAKE_COMPILER_IS_GNUCXX)
    set(WARN_FLAGS "-Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -std=c++11 ${WARN_FLAGS}")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
IF(MSVC)
    # disable some warnings:
    # C4244: conversion with possible loss of data (e.g. double to float)
    # C4996: potentially unsafe inner_product
    set(WARN_FLAGS "/W3 /wd4244 /wd4996")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FLAGS}")
ENDIF(MSVC)

set(SOURCE_FILES
    notch.hpp
    notch_io.hpp)

add_executable(demo_multilayer examples/demo_multilayer.cpp ${SOURCE_FILES})
target_include_directories(demo_multilayer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_perceptron examples/demo_perceptron.cpp ${SOURCE_FILES})
target_include_directories(demo_perceptron PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gen_twomoons examples/gen_twomoons.cpp ${SOURCE_FILES})
target_include_directories(gen_twomoons PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
