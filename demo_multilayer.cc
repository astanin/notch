/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>


#include "randomgen.hh"
#include "activation.hh"
#include "perceptron.hh"


int main(int, char *[]) {
    unique_ptr<rng_type> rng(seed_rng());
    LabeledSet trainSet {{{0,0},{-1}},
                         {{0,1},{1}},
                         {{1,0},{1}},
                         {{1,1},{-1}}};
    cout << "training set:\n" << trainSet << "\n";
    PerceptronsNetwork xorNet {2, 2, 1};
    xorNet.init(rng);
    cout << "initial NN:\n" << xorNet << "\n";
    cout << "initial out:\n";
    for (auto s : trainSet) {
        cout << s.input << " -> " << xorNet.forwardPass(s.input) << "\n";
    }

    for (int j = 0; j < 1; ++j) {
        // training cycle
        for (auto sample : trainSet) {
            auto actualOutput = xorNet.forwardPass(sample.input);
            Output err(actualOutput.size());
            for (size_t i=0; i < actualOutput.size(); ++i) {
                err[i] = sample.output[i] - actualOutput[i];
            }
            xorNet.backwardPass(err, 0.1);
        }
    }

    cout << "\nfinal NN:\n" << xorNet << "\n";
    cout << "final out:\n";
    for (auto s : trainSet) {
        cout << s.input << " -> " << xorNet.forwardPass(s.input) << "\n";
    }

    return 0;
}
