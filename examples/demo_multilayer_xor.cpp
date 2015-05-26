/// @file demo_multilayer_xor.cpp -- A multilayer perceptron for XOR problem

#include <iostream>
#include <cmath>
#include <algorithm>


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


int main(int, char *[]) {
    unique_ptr<RNG> rng(newRNG());
    LabeledDataset trainSet {{{0,0},{0}},
                         {{0,1},{1}},
                         {{1,0},{1}},
                         {{1,1},{0}}};
    LabeledDataset &testSet(trainSet);
    cout << "training set:\n" << CSVFormat(trainSet) << "\n";
    MultilayerPerceptron xorNet({2, 2, 1}, scaledTanh);
    xorNet.init(rng);
    cout << "initial NN:\n\n";
    PlainTextNetworkWriter(cout) << xorNet;
    cout << "initial out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << *xorNet.output(s.data) << "\n";
    }
    cout << "\n";

    xorNet.setLearningPolicy(0.01f, 0.9);
    trainWithSGD(xorNet, trainSet, rng, /* epochs */ 500,
                 /* callbackEvery */ 25,
                 /* callback */ [&](int i, ABackpropLayer& net) {
                     printLoss(i, net, testSet);
                });
    cout << "\n";

    cout << "final NN:\n\n";
    PlainTextNetworkWriter(cout) << xorNet;
    cout << "final out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << *xorNet.output(s.data) << "\n";
    }
    cout << "\n";

    return 0;
}
