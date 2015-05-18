cmake_minimum_required(VERSION 3.1)
project(reading-NNLM3)

set(MORE_WARNING_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -std=c++11 ${MORE_WARNING_FLAGS}")


set(SOURCE_FILES
    notch.hpp
    notch_io.hpp)

add_executable(demo_multilayer
    examples/demo_multilayer.cpp ${SOURCE_FILES})
target_include_directories(demo_multilayer
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(demo_perceptron
    examples/demo_perceptron.cpp examples/classifier.hpp ${SOURCE_FILES})
target_include_directories(demo_perceptron
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gen_twomoons
    examples/gen_twomoons.cpp ${SOURCE_FILES})
target_include_directories(gen_twomoons
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
