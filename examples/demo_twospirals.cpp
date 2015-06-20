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
	virtual int aslabel(const Output &out) { return out[0]; }
	virtual int classify(const Input &in) { return net.output(in)[0]; }
};


float meanLoss(Net &net, LabeledDataset &dataset) {
    float total;
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
    SquareAugmented SQUARE;
    trainset.apply(SQUARE);
    testset.apply(SQUARE);

    Net net = MakeNet(2 * 2) // SQUARE augmentation
        .addFC(200, scaledTanh)
        .addFC(1, scaledTanh)
        .addL2Loss()
        .init();

    net.setLearningPolicy(FixedRate(1e-3, 0.9, 1e-4));

    auto classifier = IntClassifier(net);

    SGD::train(net, trainset,
               10 /* epochs */,
               /* callbackEvery */ 1,
               /* callback */ [&](int i) {
                   auto cm = classifier.test(testset); // confusion matrix
                   cout << "epoch "
                        << setw(6) << i << " "
                        << " train loss = "
                        << setprecision(5) << setw(8)
                        << meanLoss(net, trainset)
                        << " test loss = "
                        << setprecision(5) << setw(8)
                        << meanLoss(net, testset)
                        << " test accuracy = "
                        << cm.accuracy()
                        << endl;
                   return false;
               });
}

