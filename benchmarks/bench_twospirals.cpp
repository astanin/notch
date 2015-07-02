#include <iostream>


#include "notch.hpp"
#include "notch_io.hpp"      // CSVReader


using namespace std;
using namespace notch;


float meanLoss(Net &net, LabeledDataset &dataset) {
    float total = 0.0;
    size_t n = 0;
    for (auto sample : dataset) {
        total += net.loss(sample.data, sample.label);
        n++;
    }
    return total / n;
}


int main(int argc, char *argv[]) {
    auto trainset = CSVReader().read("../data/twospirals-train.csv");
    auto testset = CSVReader().read("../data/twospirals-test.csv");

    Net net = MakeNet(trainset.inputDim())
        .addFC(50, scaledTanh)
        .addFC(50, scaledTanh)
        .addFC(50, scaledTanh)
        .addFC(50, scaledTanh)
        .addFC(10, scaledTanh)
        .addFC(1, scaledTanh)
        .addL2Loss()
        .init();

    size_t nIters = 0;
    if (argc >= 2) {
        nIters = atoi(argv[1]);
    }

    if (argc == 3 && string("--adadelta") == argv[2]) {
        cout << "# using ADADELTA\n";
        net.setLearningPolicy(AdaDelta());
    } else {
        cout << "# using fixed rate policy\n";
        net.setLearningPolicy(FixedRate(1e-3f, 0, 0));
    }

    SGD::train(net, trainset,
               nIters /* epochs */,
               EpochCallback { 1 /* print every epoch */, [&](int i) {
                   cout << "# current error = "
                        << meanLoss(net, trainset) << "\n";
                   return false;
               }});

    cout << "# test error = " << meanLoss(net, testset) << "\n"
         << "# training error = " << meanLoss(net, trainset) << "\n";
}
