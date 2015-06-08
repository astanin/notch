/// @file demo_multilayer_xor.cpp -- A multilayer perceptron for XOR problem

#include <iostream>
#include <cmath>
#include <algorithm>


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


void print_net(string tag, Net &xorNet, LabeledDataset dataset) {
    //PlainTextNetworkWriter(cout) << tag << " net:\n\n" << xorNet << "\n";
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
    unique_ptr<RNG> rng(newRNG());
    LabeledDataset dataset {{{0,0},{0}},
                            {{0,1},{1}},
                            {{1,0},{1}},
                            {{1,1},{0}}};

    Net xorNet;
    xorNet.append(MakeLayer(2, 2, scaledTanh).fc());
    xorNet.append(MakeLayer(2, 1, scaledTanh).fc());
    xorNet.append(MakeLayer(1).l2loss());
    xorNet.init(rng);

    cout << "training set:\n" << CSVFormat(dataset) << "\n";
    print_net("initial", xorNet, dataset);

    xorNet.setLearningPolicy(FixedRateWithMomentum(0.1, 0.9));
    trainWithSGD(xorNet, dataset, rng,
                 100 /* epochs */,
                 10 /* callbackPeriod */,
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
