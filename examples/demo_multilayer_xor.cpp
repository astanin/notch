/// @file demo_multilayer_xor.cpp -- A multilayer perceptron for XOR problem

#include <iostream>
#include <cmath>
#include <algorithm>


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


void print_net(string tag, MultilayerPerceptron &xorNet, LabeledDataset dataset) {
    PlainTextNetworkWriter(cout) << tag << " net:\n\n" << xorNet << "\n";
    cout << tag << " out:\n";
    for (auto s : dataset) {
        cout << s.data << " -> " << *xorNet.output(s.data) << "\n";
    }
    cout << "\n";
}


int main(int, char *[]) {
    unique_ptr<RNG> rng(newRNG());
    LabeledDataset dataset {{{0,0},{0}},
                            {{0,1},{1}},
                            {{1,0},{1}},
                            {{1,1},{0}}};
    MultilayerPerceptron xorNet({2, 2, 1}, scaledTanh);
    xorNet.init(rng);

    cout << "training set:\n" << CSVFormat(dataset) << "\n";
    print_net("initial", xorNet, dataset);

    xorNet.setLearningPolicy(FixedRateWithMomentum(0.01, 0.9));
    trainWithSGD(xorNet, dataset, rng,
                 500 /* epochs */,
                 100 /* callbackPeriod */,
                 /* callback */ [&](int i) {
                    cout << "epoch " << i << " total loss = "
                         << totalLoss(L2_loss, xorNet, dataset) << "\n";
                    return false; // don't terminate;
                });
    cout << "\n";

    print_net("final", xorNet, dataset);
    return 0;
}
