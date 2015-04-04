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

Demo: [rosenblatt.cc](rosenblatt.cc).
