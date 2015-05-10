/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>
#include <cmath>
#include <algorithm>


#include "randomgen.hh"
#include "activation.hh"
#include "perceptron.hh"


double loss(PerceptronsNetwork &net, LabeledSet &testSet) {
    double loss = 0.0;
    for (auto sample : testSet) {
        auto correct = sample.output;
        auto result = net.forwardPass(sample.input);
        transform(correct.begin(), correct.end(),
                  result.begin(),
                  result.begin(),
                  [](double y_c, double y) { return abs(y_c - y); });
        for (auto val: result) {
            loss += val;
        }
    }
    return loss;
}


int main(int, char *[]) {
    unique_ptr<rng_type> rng(seed_rng());
    LabeledSet trainSet {{{0,0},{0}},
                         {{0,1},{1}},
                         {{1,0},{1}},
                         {{1,1},{0}}};
    LabeledSet &testSet(trainSet);
    cout << "training set:\n" << trainSet << "\n";
    PerceptronsNetwork xorNet {2, 2, 1};
    xorNet.init(rng);
    cout << "initial NN:\n" << xorNet << "\n";
    cout << "initial out:\n";
    for (auto s : trainSet) {
        cout << s.input << " -> " << xorNet.forwardPass(s.input) << "\n";
    }
    cout << "initial loss: " << loss(xorNet, testSet) << "\n";

    for (int j = 0; j < 20000; ++j) {
        // training cycle
        for (auto sample : trainSet) {
            //cout << "in: " << sample.input << "\n";
            auto actualOutput = xorNet.forwardPass(sample.input);
            Output err(actualOutput.size());
            for (size_t i=0; i < actualOutput.size(); ++i) {
                err[i] = sample.output[i] - actualOutput[i];
            }
            xorNet.backwardPass(err, 0.01);
        }
        if (j % 500 == 0) {
            cout << "epoch " << j+1 << " loss: " << loss(xorNet, testSet) << "\n";
        }
    }

    cout << "\nfinal NN:\n" << xorNet << "\n";
    cout << "final out:\n";
    for (auto s : trainSet) {
        cout << s.input << " -> " << xorNet.forwardPass(s.input) << "\n";
    }

    return 0;
}
