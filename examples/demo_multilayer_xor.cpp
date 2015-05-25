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
    cout << "initial NN:\n" << xorNet << "\n";
    cout << "initial out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << *xorNet.output(s.data) << "\n";
    }
    cout << "\n";

    cout << "initial loss: " << totalLoss(L2_loss, xorNet, testSet) << "\n";

    xorNet.setLearningPolicy(0.01f);
    for (int j = 0; j < 5000; ++j) {
        // training cycle
        trainSet.shuffle(rng);
        for (auto sample : trainSet) {
            auto out_ptr = xorNet.output(sample.data);
            Array err = sample.label - *out_ptr;
            xorNet.backprop(err);
            xorNet.update();
        }
        if (j % 500 == 0) {
            cout << "epoch " << j+1 << " loss: " << totalLoss(L2_loss, xorNet, testSet) << "\n";
        }
    }
    cout << "\n";

    cout << "final NN:\n" << xorNet << "\n";
    cout << "final out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << *xorNet.output(s.data) << "\n";
    }
    cout << "\n";

    return 0;
}
