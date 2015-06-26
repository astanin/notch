/// @file demo_io.cpp -- Save and load neural network parameters.

#include <iostream>
#include <sstream>

#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;
using namespace notch;


int main() {

    // create a multilayer perceptron and save it to a string stream
    Net mlp = MakeNet().MultilayerPerceptron({2, 2, 1}).addL2Loss().init();
    stringstream ss;
    PlainTextNetworkWriter(ss) << mlp;

    // show original
    cout << "saved:\n\n```\n";
    cout << ss.str();
    cout << "```\n\n";

    // read a copy of the multilayer perceptron from the string stream
    ss.seekg(0);
    Net mlp_copy = PlainTextNetworkReader(ss).read();

    // show a copy
    cout << "loaded:\n\n```\n";
    PlainTextNetworkWriter(cout) << mlp_copy;
    cout << "```\n\n";
}
