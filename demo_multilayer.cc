/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>


#include "randomgen.hh"
#include "activation.hh"
#include "perceptron.hh"


int main(int, char *[]) {
    unique_ptr<rng_type> rng(seed_rng());
    PerceptronsLayer layer(2, 2);
    layer.init(rng);
    Input in{-1.0, 1.0};
    cout << "weights:\n";
    cout << layer << "\n";
    cout << "input:\n";
    cout << in << "\n";
    auto out = layer.forwardPass(in);
    cout << "output:\n";
    cout << out << "\n";
    Output label{0.0, 0.0};
    Output error(2);
    for (int i = 0; i < 2; ++i) {
        error[i] = label[i] - out[i];
    }
    cout << "error:\n";
    cout << error << "\n";
    auto bpError = layer.backwardPass(in, error, 0.25);
    cout << "propagated error:\n";
    cout << bpError << "\n";
    cout << "new weights after BP:\n";
    cout << layer << "\n";
    out = layer.forwardPass(in);
    cout << "new output after BP:\n";
    cout << out << "\n";
    for (int i = 0; i < 2; ++i) {
        error[i] = label[i] - out[i];
    }
    cout << "new error after BP:\n";
    cout << error << "\n";
    return 0;
}
