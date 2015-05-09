/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>


#include "randomgen.hh"
#include "activation.hh"
#include "perceptron.hh"


int main(int, char *[]) {
    unique_ptr<rng_type> rng(seed_rng());
    LabeledSet trainSet {{{0,0},{0}},
                         {{0,1},{1}},
                         {{1,0},{1}},
                         {{1,1},{0}}};
    cout << "training set:\n" << trainSet << "\n";
    PerceptronsNetwork xorNet {2, 2, 1};
    xorNet.init(rng);
    cout << "initial NN:\n" << xorNet << "\n";
    return 0;
}
