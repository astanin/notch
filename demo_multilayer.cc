/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>


#include "randomgen.hh"
#include "activation.hh"
#include "perceptron.hh"


int main(int, char *[]) {
    unique_ptr<rng_type> rng(seed_rng());
    PerceptronsLayer layer(2, 2);
    layer.init(rng, 0.1);
    Input in{10, 100};
    cout << "weights:\n";
    cout << layer << "\n";
    cout << "input:\n";
    cout << in << "\n";
    auto out = layer.forwardPass(in);
    cout << "output:\n";
    cout << out << "\n";
    return 0;
}
