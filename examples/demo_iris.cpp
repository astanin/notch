/// @file demo_iris.cpp -- A multilayer perceptron on Iris dataset
///
/// usage: demo_iris [path_to_data (default: ../data/iris.csv)]

#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include <iomanip>


#include "notch.hpp"
#include "notch_io.hpp"
#include "notch_pre.hpp"
#include "notch_metrics.hpp"


using namespace std;
using namespace notch;


ostream &operator<<(ostream &out, const Dataset &d) {
    for (auto v : d) {
        out << v << "\n";
    }
    return out;
}

class IntClassifier : public AClassifier < int, 1 > {
private:
	OneHotEncoder &enc;
	Net &net;

public:
	IntClassifier(Net &net, OneHotEncoder &enc)
		: enc(enc), net(net) {}

	virtual int aslabel(const Output &out) {
		return enc.unapply(out)[0];
	}
	virtual int classify(const Input &in) {
		return enc.unapply(net.output(in))[0];
	}
};

float meanLoss(Net &net, LabeledDataset &dataset) {
    float total = 0.0;
    size_t n = 0;
    for (auto sample : dataset) {
        total += net.loss(sample.data, sample.label);
        n++;
    }
    return total / n;
}

void print_metrics(IntClassifier &classifier, LabeledDataset &dataset) {
    auto cm = classifier.test(dataset); // cm is a confusion matrix
    cout << "F1(0) = " << setprecision(2) << setw(4) << cm.F1score(0)
        << " F1(1) = " << setprecision(2) << setw(4) << cm.F1score(1)
        << " F1(2) = " << setprecision(2) << setw(4) << cm.F1score(2)
        << " accuracy = " << cm.accuracy()
        << endl;
}


int main(int argc, char *argv[]) {
    string csvFile("../data/iris.csv");
    if (argc == 2) {
        csvFile = string(argv[1]);
    }
    cout << "reading dataset from CSV " << csvFile << "\n";
    ifstream f(csvFile);
    if (!f.is_open()) {
        cerr << "cannot open " << csvFile << "\n";
        exit(-1);
    }

    LabeledDataset irisData = CSVReader().read(f);
    OneHotEncoder labelEnc(irisData.getLabels());
    irisData.applyToLabels(labelEnc);

    Net net = MakeNet()
        .setInputDim(irisData.inputDim())
        .addFC(24, scaledTanh)
        .addFC(3, linearActivation)
        .addSoftmax()
        .make();

    unique_ptr<RNG> rng(Init::newRNG());
    net.init(rng, Init::uniformNguyenWidrow);

    IntClassifier classifier(net, labelEnc);

    net.setLearningPolicy(FixedRate(0.0001 /* rate */, 0.9 /* momentum */));
    SGD::train(rng, net, irisData, 64 /* epochs */,
               EpochCallback {/* every */ 4 /* epochs */,
               /* callback */ [&](int i) {
                   cout << "epoch "
                        << setw(4) << i << " "
                        << "loss = "
                        << setprecision(3) << setw(6)
                        << meanLoss(net, irisData) << " ";
                   print_metrics(classifier, irisData);
                   return false;
               }});
    cout << "\n";

    ofstream nnfile("demo_iris_network.txt");
    if (nnfile.is_open()) {
        PlainTextNetworkWriter(nnfile) << net << "\n";
        cout << "wrote demo_iris_network.txt\n";
    }
}
