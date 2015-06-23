#include <iostream>


using namespace std;


#include "notch.hpp"
#include "notch_io.hpp"      // CSVReader
#include "notch_metrics.hpp" // AClassifier, ConfusionMatrix
#include "notch_pre.hpp"     // SquareAugmented


class IntClassifier : public AClassifier < int, 0 > {
private:
	Net &net;
public:
	IntClassifier(Net &net) : net(net) {}
	virtual int aslabel(const Output &out) { return out[0] > 0; }
	virtual int classify(const Input &in) { return net.output(in)[0] > 0; }
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


int main() {
    auto trainset = CSVReader().read("../data/twospirals-train.csv");
    auto testset = CSVReader().read("../data/twospirals-test.csv");

    // with SQUARE: accuracy = 99.5% (130000 epochs on 4-80-20-1 NN, ~5 min)
    // without:     accuracy = 99.5% (50000 epochs on 2-50-50-50-50-10-1 NN, ~7 min)
    SquareAugmented SQUARE;
    trainset.apply(SQUARE);
    testset.apply(SQUARE);

    Net net = MakeNet(trainset.inputDim())
        .addFC(80, scaledTanh)
        .addFC(20, scaledTanh)
        .addFC(1, scaledTanh)
        .addL2Loss()
        .init();

    net.setLearningPolicy(AdaDelta());

    auto classifier = IntClassifier(net);

    SGD::train(net, trainset,
               130000 /* epochs */,
               EpochCallback { 10000, [&](int i) {
                   auto cm = classifier.test(testset); // confusion matrix
                   cout << "epoch "
                        << setw(6) << i << " "
                        << " train loss = "
                        << setprecision(3) << setw(5) << showpoint
                        << meanLoss(net, trainset)
                        << " test loss = "
                        << setprecision(3) << setw(5) << showpoint
                        << meanLoss(net, testset)
                        << " accuracy = "
                        << setprecision(3) << setw(5) << showpoint
                        << cm.accuracy()
                        << " F1 = "
                        << setprecision(3) << setw(5) << showpoint
                        << cm.F1score()
                        << endl;
                   return false;
               }});

    string dumpfile = "demo_twospirals_network.txt";
    ofstream nnfile(dumpfile);
    if (nnfile.is_open()) {
        PlainTextNetworkWriter(nnfile) << net << "\n";
        cout << "wrote " << dumpfile << "\n";
    }
}

