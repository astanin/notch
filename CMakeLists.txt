cmake_minimum_required(VERSION 3.0)
project(notch)
enable_testing()

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
# GCC workaround for MSYS2 and Cygwin
# see http://stackoverflow.com/q/28997206/25450
IF(CMAKE_COMPILER_IS_GNUCXX)
    IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        IF(${MINGW})
            message(STATUS "Detected MinGW")
        ELSE(${MINGW})
            message(STATUS "Enabling workaround for GCC on CYGWIN/MSYS2")
            add_definitions(-DENABLE_GCC49_CYGWIN_MSYS2_WORKAROUND)
        ENDIF(${MINGW})
    ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

IF(USE_CBLAS)
    find_library(CBLAS_LIBRARY NAMES openblas cblas)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATHS /usr/include/openblas)
    message(STATUS "CBLAS_LIBRARY: " ${CBLAS_LIBRARY})
    message(STATUS "CBLAS_INCLUDE_DIR: " ${CBLAS_INCLUDE_DIR})
    IF(CBLAS_LIBRARY)
        set(CBLAS_FOUND ON)
        message(STATUS "Found CBLAS")
        add_definitions(-DNOTCH_USE_CBLAS)
        set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
        set(CBLAS_INCLUDE_DIRS ${CBLAS_INCLUDE_DIR})
        include_directories(${CBLAS_INCLUDE_DIRS})
    ENDIF(CBLAS_LIBRARY)
ENDIF(USE_CBLAS)

set(SOURCE_FILES
    notch.hpp
    notch_io.hpp
    notch_pre.hpp
    notch_metrics.hpp)

add_test(test_notch test_notch "[core]")
add_test(test_notch_io test_notch "[io]")
add_executable(test_notch
               test/test_main.cpp test/test_notch.cpp test/test_notch_io.cpp
               test/test_notch_pre.cpp
               ${SOURCE_FILES})
target_include_directories(test_notch PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_xor examples/demo_xor.cpp ${SOURCE_FILES})
target_include_directories(demo_xor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_iris examples/demo_iris.cpp ${SOURCE_FILES})
target_include_directories(demo_iris PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_mnist examples/demo_mnist.cpp ${SOURCE_FILES})
target_include_directories(demo_mnist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_twospirals examples/demo_twospirals.cpp ${SOURCE_FILES})
target_include_directories(demo_twospirals PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_io examples/demo_io.cpp ${SOURCE_FILES})
target_include_directories(demo_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

IF(CBLAS_FOUND)
    target_link_libraries(test_notch ${CBLAS_LIBRARIES})
    target_link_libraries(demo_xor ${CBLAS_LIBRARIES})
    target_link_libraries(demo_iris ${CBLAS_LIBRARIES})
    target_link_libraries(demo_mnist ${CBLAS_LIBRARIES})
    target_link_libraries(demo_twospirals ${CBLAS_LIBRARIES})
    target_link_libraries(demo_io ${CBLAS_LIBRARIES})
ELSE(CBLAS_FOUND)
    message(STATUS "Not using CBLAS")
ENDIF(CBLAS_FOUND)
