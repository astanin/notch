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

    Perceptron p(trainset.getInputSize());
    cout << "Initial weights:                     " << p.fmt() << "\n";
    ConfusionMatrix cm1 = p.test(testset);
    cout << cm1.truePositives << " " << cm1.falsePositives << "\n";
    cout << cm1.falseNegatives << " " << cm1.trueNegatives << "\n";
    cout << "Initial accuracy:                    " << cm1.accuracy() << "\n";

    p.trainConverge(trainset);
    cout << "Weights after convergence training:  " << p.fmt() << "\n";
    ConfusionMatrix cm2 = p.test(testset);
    cout << cm2.truePositives << " " << cm2.falsePositives << "\n";
    cout << cm2.falseNegatives << " " << cm2.trueNegatives << "\n";
    cout << "Accuracy after convergence training: " << cm2.accuracy() << "\n";

    cout << "\nMisclassified set:\n";
    LabeledSet::LabeledPairPredicate isMisclassified = [&p](const Input &in, const Output &out) {
        return p.response(in)*out[0] < 0;
    };
    auto misclassifiedSet = testset.filter(isMisclassified);
    cout << misclassifiedSet;
}

