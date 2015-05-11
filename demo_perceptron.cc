// Demo for Chapter 1. Rosenblatt's Perceptron.
#include <iostream>
#include <fstream>


#include "dataset.hh"
#include "perceptron.hh"


using namespace std;
int N_ITERS = 100;


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: demo_perceptron train.data test.data\n";
        exit(-1);
    }
    ifstream trainIn(argv[1]);
    ifstream testIn(argv[2]);
    LabeledSet trainset = LabeledSet(trainIn);
    LabeledSet testset = LabeledSet(testIn);

    StandalonePerceptron p(trainset.inputDim());
    cout << "Initial weights:                     " << p.fmt() << "\n";
    ConfusionMatrix cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives
         << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives
         << "\n";
    cout << "Initial accuracy:                    " << cm.accuracy() << "\n";

    cout << "\n1 iteration...\n\n";
    p.trainConverge(trainset, 1);
    cout << "Weights after convergence training:  " << p.fmt() << "\n";
    cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives
         << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives
         << "\n";
    cout << "Accuracy after convergence training: " << cm.accuracy() << "\n";

    cout << "\n" << N_ITERS - 1 << " more iterations...\n\n";
    p.trainConverge(trainset, N_ITERS - 1);

    cout << "Weights after convergence training:  " << p.fmt() << "\n";
    cm = p.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives
         << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives
         << "\n";
    cout << "Accuracy after convergence training: " << cm.accuracy() << "\n";

    cout << "\nvs " << N_ITERS << " iterations of batch training...\n\n";
    StandalonePerceptron p2(trainset.inputDim());
    p2.trainBatch(trainset, N_ITERS);
    cout << "Weights after batch training:        " << p2.fmt() << "\n";
    cm = p2.test(testset);
    cout << "Confusion matrix:\n";
    cout << "    TP=" << cm.truePositives << " FP=" << cm.falsePositives
         << "\n";
    cout << "    FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives
         << "\n";
    cout << "Accuracy after batch training:       " << cm.accuracy() << "\n";
}

