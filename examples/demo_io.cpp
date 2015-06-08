/// @file demo_io.cpp -- Save and load neural network parameters.

#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"

using namespace std;

int main() {
#if 0
    auto rng = newRNG();
    stringstream ss;
    MultilayerPerceptron mlp {2, 2, 1};
    mlp.init(rng);
    PlainTextNetworkWriter(ss) << mlp;

    cout << "saved:\n\n```\n";
    cout << ss.str();
    cout << "```\n\n";

    ss.seekg(0);
    MultilayerPerceptron mlp_copy;
    PlainTextNetworkReader(ss) >> mlp_copy;

    cout << "loaded:\n\n```\n";
    PlainTextNetworkWriter(cout) << mlp_copy;
    cout << "```\n\n";
#endif
}
