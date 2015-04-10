// Demo for Chapter 1. Rosenblatt's Perceptron.
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
    ConfusionMatrix cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives << "\n";
    cout << "Initial accuracy:                    " << cm.accuracy() << "\n";

    cout << "\n1 iteration...\n\n";
    p.trainConverge(trainset);
    cout << "Weights after convergence training:  " << p.fmt() << "\n";
    cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives << "\n";
    cout << "Accuracy after convergence training: " << cm.accuracy() << "\n";

    cout << "\n99 more iterations...\n\n";
    for (int i=0; i<99; i++) {
        p.trainConverge(trainset);
    }
    cout << "Weights after convergence training:  " << p.fmt() << "\n";
    cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives << "\n";
    cout << "Accuracy after convergence training: " << cm.accuracy() << "\n";

    cout << "\nvs 100 iterations of batch training...\n\n";
    Perceptron p2(trainset.getInputSize());
    p2.trainBatch(trainset, 100, [](int epoch) { return 1.0/(1+epoch); });
    cout << "Weights after batch training:        " << p.fmt() << "\n";
    cm = p2.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives << "\n";
    cout << "Accuracy after batch training:       " << cm.accuracy() << "\n";

}

