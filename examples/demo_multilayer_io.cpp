/// @file demo_multilayer_io.cpp -- Saving and loading multilayer perceptron parameters

#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

int main() {
    auto rng = newRNG();
    stringstream ss;
    MultilayerPerceptron mlp {2, 2, 1};
    mlp.init(rng);
    PlainTextNetworkWriter(ss) << mlp;

    cout << "saved:\n\n```\n";
    cout << ss.str();
    cout << "```\n\n";
}
