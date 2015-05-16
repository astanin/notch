cmake_minimum_required(VERSION 3.1)
project(reading-NNLM3)

set(MORE_WARNING_FLAGS "-Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm -std=c++11 ${MORE_WARNING_FLAGS}")

set(SOURCE_FILES
    classifier.hpp
    nevromancer.hpp)

add_executable(demo_perceptron demo_perceptron.cpp ${SOURCE_FILES})
add_executable(demo_multilayer demo_multilayer.cpp ${SOURCE_FILES})
add_executable(gen_twomoons gen_twomoons.cpp ${SOURCE_FILES})
