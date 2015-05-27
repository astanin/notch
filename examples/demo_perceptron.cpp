/// @file demo_perceptron.cpp -- Rosenblatt's Perceptron
#include <iostream>
#include <fstream>
#include <stdexcept>


#include "notch.hpp"
#include "notch_io.hpp"
#include "notch_metrics.hpp"


using namespace std;
int N_ITERS = 100;


class LinearPerceptronClassifier : public AClassifier<bool, true> {
private:
	LinearPerceptron &perceptron;

public:
	LinearPerceptronClassifier(LinearPerceptron &perceptron)
		: perceptron(perceptron) {}

	virtual bool aslabel(const Output &y) {
		return (y[0] > 0);
	}

	virtual bool classify(const Input &x) {
		return (perceptron.output(x) > 0);
	}
};


void print_stats(LinearPerceptron &p, const LabeledDataset &testSet) {
    LinearPerceptronClassifier lpc(p);
    ConfusionMatrix<bool, true> cm = lpc.test(testSet);
    cout << "Synaptic weights: " << p.getWeights() << "\n";
    cout << "Precision:  " << cm.precision() << "\n";
    cout << "Recall:     " << cm.recall() << "\n";
    cout << "Accuracy:   " << cm.accuracy() << "\n";
}

int main(int argc, char *argv[]) {
    if ((argc != 1 && argc != 3) || (argc > 1 && string("--help") == argv[1])) {
        cerr << "Usage: demo_perceptron [train.data test.data]\n";
        exit(-1);
    }
    string trainFile("../data/twomoons-train.fann");
    string testFile("../data/twomoons-test.fann");
    if (argc == 3) {
        trainFile = argv[1];
        testFile = argv[2];
    }
    cout << "Loading " << trainFile << " and " << testFile << "\n";
    LabeledDataset trainSet;
    LabeledDataset testSet;
    try {
        trainSet = (FANNReader::read(trainFile));
        testSet = (FANNReader::read(testFile));
    } catch (runtime_error &e) {
        cout << e.what() << "\n";
        exit(-2);
    }

    LinearPerceptron p(trainSet.inputDim());

    cout << "Start:\n\n";
    print_stats(p, testSet);

    cout << "\nAfter " << N_ITERS << " iterations of convergence training:\n\n";
    trainPerceptron(p, trainSet, N_ITERS, 0.1f);
    print_stats(p, testSet);

    LinearPerceptron p2(trainSet.inputDim());
    cout << "\nvs after " << N_ITERS << " iterations of batch training...\n\n";
    batchTrainPerceptron(p2, trainSet, N_ITERS, 0.1f);
    print_stats(p2, testSet);
}

