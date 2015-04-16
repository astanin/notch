cmake_minimum_required(VERSION 3.1)
project(reading-NNLM3)

set(ROMPIPALLE_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -std=c++11 ${ROMPIPALLE_FLAGS}")

set(SOURCE_FILES
    activation.hh
    classifier.hh
    dataset.hh
    perceptron.hh)

add_executable(demo_perceptron demo_perceptron.cc ${SOURCE_FILES})
add_executable(demo_multilayer demo_multilayer.cc ${SOURCE_FILES})
add_executable(gen_twomoons gen_twomoons.cc ${SOURCE_FILES})
