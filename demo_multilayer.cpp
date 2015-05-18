/// Demo for Chapter 4. Multilayer Perceptrons
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
    cout << "training set:\n" << trainSet << "\n";
    MultilayerPerceptron xorNet({2, 2, 1}, scaledTanh);
    xorNet.init(rng);
    cout << "initial NN:\n" << xorNet << "\n";
    cout << "initial out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << xorNet.forwardPass(s.data) << "\n";
    }
    cout << "\n";

    cout << "initial loss: " << totalLoss(L2_loss, xorNet, testSet) << "\n";

    for (int j = 0; j < 5000; ++j) {
        // training cycle
        for (auto sample : trainSet) {
            auto actualOutput = xorNet.forwardPass(sample.data);
            Output err(actualOutput.size());
            for (size_t i=0; i < actualOutput.size(); ++i) {
                err[i] = sample.label[i] - actualOutput[i];
            }
            xorNet.backwardPass(err, 0.01);
        }
        if (j % 500 == 0) {
            cout << "epoch " << j+1 << " loss: " << totalLoss(L2_loss, xorNet, testSet) << "\n";
        }
    }
    cout << "\n";

    cout << "final NN:\n" << xorNet << "\n";
    cout << "final out:\n";
    for (auto s : trainSet) {
        cout << s.data << " -> " << xorNet.forwardPass(s.data) << "\n";
    }
    cout << "\n";

    return 0;
}
