/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>
#include <cmath>
#include <algorithm>


#include "nevromancer.hpp"


using namespace std;


double loss(MultilayerPerceptron &net, LabeledDataset &testSet) {
    double loss = 0.0;
    for (auto sample : testSet) {
        auto correct = sample.label;
        auto result = net.forwardPass(sample.data);
        transform(begin(correct), end(correct),
                  begin(result),
                  begin(result),
                  [](double y_c, double y) { return abs(y_c - y); });
        for (auto val: result) {
            loss += val;
        }
    }
    return loss;
}


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

    cout << "initial loss: " << loss(xorNet, testSet) << "\n";

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
            cout << "epoch " << j+1 << " loss: " << loss(xorNet, testSet) << "\n";
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
