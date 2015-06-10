/// @file demo_xor.cpp -- A multilayer perceptron for XOR problem

#include <iostream>
#include <cmath>
#include <algorithm>


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


void print_net(string tag, Net &xorNet, LabeledDataset dataset) {
    PlainTextNetworkWriter(cout) << tag << " net:\n\n" << xorNet << "\n";
    cout << tag << " out:\n";
    for (auto s : dataset) {
        auto out = xorNet.output(s.data);
        cout << s.data << " -> " << out << "\n";
    }
    cout << "\n";
}

float meanLoss(Net &net, LabeledDataset &dataset) {
    float total = 0.0;
    int n = 0;
    for (auto sample : dataset) {
        total += net.loss(sample.data, sample.label);
        n++;
    }
    return total / n;
}


int main(int, char *[]) {
    LabeledDataset dataset {{{0,0},{0}},
                            {{0,1},{1}},
                            {{1,0},{1}},
                            {{1,1},{0}}};

    Net xorNet = MakeNet()
        .MultilayerPerceptron({2, 2, 1}, scaledTanh)
        .addL2Loss().init();

    cout << "training set:\n" << CSVFormat(dataset) << "\n";
    print_net("initial", xorNet, dataset);

    xorNet.setLearningPolicy(FixedRate(0.01 /* rate */, 0.9 /* momentum */));
    SGD::train(xorNet, dataset,
               500 /* epochs */,
               100 /* callbackPeriod */,
               [&](int i) { /* callback */
                   float loss = meanLoss(xorNet, dataset);
                   cout << "epoch " << setw(3) << right << i
                        << " loss = " << setw(10) << left << loss
                        << endl;
                   return false; // don't terminate;
               });
    cout << "\n";

    print_net("final", xorNet, dataset);
    return 0;
}
