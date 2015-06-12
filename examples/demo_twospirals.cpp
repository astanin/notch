#include <iostream>


using namespace std;


#include "notch.hpp"
#include "notch_io.hpp"      // FANNReader
#include "notch_metrics.hpp" // AClassifier, ConfusionMatrix


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
    auto trainset = FANNReader::read("../data/twospirals-train.fann");
    auto testset = FANNReader::read("../data/twospirals-test.fann");

    Net net = MakeNet(2)
        .addFC(5, scaledTanh)
        .addFC(5, scaledTanh)
        .addFC(1, scaledTanh)
        .addHingeLoss()
        .init(Init::uniformNguyenWidrow);

    net.setLearningPolicy(FixedRate(0.001, 0.9, 0.00001));

    auto classifier = IntClassifier(net);

    SGD::train(net, trainset,
               5000 /* epochs */,
               /* callbackEvery */ 1000,
               /* callback */ [&](int i) {
                   auto cm = classifier.test(testset); // confusion matrix
                   cout << "epoch "
                        << setw(4) << i << " "
                        << setprecision(5) << setw(8)
                        << " train loss = "
                        << meanLoss(net, trainset)
                        << " test loss = "
                        << meanLoss(net, testset)
                        << " test accuracy = "
                        << cm.accuracy()
                        << endl;
                   for (size_t l = 0; l < net.size(); ++l) {
                        auto lptr = net.getLayer(l);
                        if (!lptr) {
                            cerr << "  layer " << l + 1 << " is nullptr\n";
                            continue;
                        }
                        auto &lref = const_cast<ABackpropLayer&>(*lptr);
                        auto outptr = lref.getOutputBuffer();
                        if (!outptr) {
                            cerr << "  layer " << l + 1 << " outputBuffer is nullptr\n";
                            continue;
                        }
                        cout << "  layer " << l + 1 << " out: " << *outptr << "\n";
                   }
                   return false;
               });
}

