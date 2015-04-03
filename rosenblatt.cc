// Chapter 1. Rosenblatt's Perceptron.
// Section 1.6. The Batch Perceptron Algorithm.
#include <iostream>
#include <fstream>


#include "dataset.hh"
#include "perceptron.hh"


using namespace std;


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: rosenblatt train.data test.data\n";
        exit(-1);
    }
    ifstream trainIn(argv[1]);
    ifstream testIn(argv[2]);
    LabeledSet trainset = LabeledSet(trainIn);
    LabeledSet testset = LabeledSet(testIn);

    Perceptron p(trainset.inputSize);
    cout << "Initial weights: " << p.fmt() << "\n";
    p.trainPCA(trainset);
    cout << "Final weights: " << p.fmt() << "\n";

    int nErrors = 0;
    for (int i=0; i<testset.nSamples; ++i) {
        auto x = testset.inputs[i];
        double y_true = testset.outputs[i][0];
        double y_predicted = p.response(x);
        if (y_true * y_predicted < 0) {
            nErrors++;
            for (auto x_j : x) {
                cout << x_j << " ";
            }
            cout << "=> " << y_predicted;
            cout << " ERROR\n";
        }
    }
    cout << "Accuracy: " << (100.0*(1-nErrors*1.0/testset.nSamples)) << "%\n";
}

