// Demo for Chapter 1. Rosenblatt's Perceptron.
#include <iostream>
#include <fstream>


#include "notch.hpp"
#include "notch_io.hpp"
#include "classifier.hpp"


using namespace std;
int N_ITERS = 100;


void print_stats(LinearPerceptron &p, const LabeledDataset &testSet) {
    LinearPerceptronClassifier lpc(p);
    ConfusionMatrix cm = lpc.test(testSet);
    cout << "Synaptic weights:\n";
    cout << "  " << p << "\n";
    cout << "Confusion matrix:\n";
    cout << "  TP=" << cm.truePositives << " FP=" << cm.falsePositives << "\n";
    cout << "  FN=" << cm.falseNegatives << " TN=" << cm.trueNegatives << "\n";
    cout << "Accuracy:\n  " << cm.accuracy() << "\n";
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: demo_perceptron train.data test.data\n";
        exit(-1);
    }

    LabeledDataset trainSet(FANNReader::read(argv[1]));
    LabeledDataset testSet(FANNReader::read(argv[2]));

    LinearPerceptron p(trainSet.inputDim());

    cout << "Start:\n\n";
    print_stats(p, testSet);

    cout << "\nAfter " << N_ITERS << " iterations of convergence training:\n\n";
    trainConverge(p, trainSet, N_ITERS, 0.1);
    print_stats(p, testSet);

    LinearPerceptron p2(trainSet.inputDim());
    cout << "\nvs after " << N_ITERS << " iterations of batch training...\n\n";
    trainBatch(p2, trainSet, N_ITERS, 0.1);
    print_stats(p2, testSet);
}

