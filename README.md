Method Examples for Neural Networks and Learning Machines 3rd Ed.
=================================================================

"Learning through doing". Implementing methods from "Neural Networks and
Learning Machines, 3rd Ed." to understand them better.

Implementation language: C++11.

Chapter 1. Rosenblatt's Perceptron
----------------------------------

Implemetation: [perceptron.hh](perceptron.hh).

Two training methods:

 * perceptron convergence training (`Perceptron::trainConverge()`)
 * batch training on misclassified subsets (`Perceptron::trainBatch()`)

Demo: [demo_perceptron.cc](demo_perceptron.cc).



How to build
------------

### On Windows

  * Install MinGW and CMake
  * `mkdir build`
  * `cd build`
  * `cmake.exe -G "MinGW Makefiles" ..`

    or

    `cmake.exe -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug ..`
  * `mingw32-make`
