/// Demo for Chapter 4. Multilayer Perceptrons
#include <iostream>


#include "activation.hh"
#include "perceptron.hh"


int main(int , char*[]) {
    PerceptronsLayer layer(2, 2);
    Input in { 10, 20 };
    cout << "weights:\n";
    cout << layer << "\n";
    cout << "input:\n";
    cout << in << "\n";
    auto out = layer.forwardPass(in);
    cout << "output:\n";
    cout << out << "\n";
    return 0;

}
