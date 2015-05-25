cmake_minimum_required(VERSION 3.0)
project(notch)

IF(CMAKE_COMPILER_IS_GNUCXX)
    set(WARN_FLAGS "-Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -std=c++11 ${WARN_FLAGS}")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)
IF(MSVC)
    # disable some warnings:
    # C4244: conversion with possible loss of data (e.g. double to float)
    # C4290: C++ exception specification is ignored
    # C4996: potentially unsafe inner_product
    set(WARN_FLAGS "/W3 /wd4244 /wd4290 /wd4996")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARN_FLAGS}")
ENDIF(MSVC)

IF(USE_CBLAS)
    find_library(CBLAS_LIBRARY NAMES openblas cblas)
    find_path(CBLAS_INCLUDE_DIR cblas.h)
    IF(CBLAS_LIBRARY)
        set(CBLAS_FOUND ON)
        message(STATUS "Found CBLAS " ${CBLAS_LIBRARY})
        add_definitions(-DNOTCH_USE_CBLAS)
        set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
        set(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})
        include_directories(${CBLAS_INCLUDE_DIRS})
    ENDIF(CBLAS_LIBRARY)
ENDIF(USE_CBLAS)

set(SOURCE_FILES
    notch.hpp
    notch_io.hpp
    notch_pre.hpp)

add_executable(gen_twomoons examples/gen_twomoons.cpp ${SOURCE_FILES})
target_include_directories(gen_twomoons PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_perceptron examples/demo_perceptron.cpp ${SOURCE_FILES})
target_include_directories(demo_perceptron PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_multilayer_xor examples/demo_multilayer_xor.cpp ${SOURCE_FILES})
target_include_directories(demo_multilayer_xor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_multilayer_iris examples/demo_multilayer_iris.cpp ${SOURCE_FILES})
target_include_directories(demo_multilayer_iris PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

IF(CBLAS_FOUND)
    target_link_libraries(gen_twomoons ${CBLAS_LIBRARIES})
    target_link_libraries(demo_perceptron ${CBLAS_LIBRARIES})
    target_link_libraries(demo_multilayer_xor ${CBLAS_LIBRARIES})
    target_link_libraries(demo_multilayer_iris ${CBLAS_LIBRARIES})
ELSE(CBLAS_FOUND)
    message(STATUS "Not using CBLAS")
ENDIF(CBLAS_FOUND)
