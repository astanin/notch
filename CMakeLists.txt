cmake_minimum_required(VERSION 3.1)
project(reading-NNLM3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    classifier.hh
    dataset.hh
    perceptron.hh)

add_executable(demo_perceptron demo_perceptron.cc ${SOURCE_FILES})
add_executable(gen_twomoon gen_twomoon.cc ${SOURCE_FILES})
