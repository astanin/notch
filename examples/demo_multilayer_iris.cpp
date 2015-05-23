/// @file demo_multilayer_iris.cpp -- A multilayer perceptron on Iris dataset
///
/// usage: demo_multilayer_iris [path_to_data (default: ../data/iris.csv)]

#include <iostream>
#include <fstream>
#include <ostream>
#include <string>
#include <iomanip>


#include "notch.hpp"
#include "notch_io.hpp"
#include "notch_pre.hpp"


using namespace std;


ostream &operator<<(ostream &out, const Dataset &d) {
    for (auto v : d) {
        out << v << "\n";
    }
    return out;
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

    LabeledDataset trainSet = CSVReader<>::read(f);
    OneHotEncoder labelEnc(trainSet.getLabels());
    trainSet.transformLabels(labelEnc);
    //cout << ArrowFormat(trainSet);

    MultilayerPerceptron net({4, 6, 3}, scaledTanh);
    unique_ptr<RNG> rng(newRNG());
    net.init(rng);
    cout << net << "\n\n";

    cout << "initial loss: " << totalLoss(L2_loss, net, trainSet) << "\n";

    float learningRate = 0.01f;
    for (int j = 0; j < 1000; ++j) {
        // training cycle
        for (auto sample : trainSet) {
            Array actualOutput = *net.output(sample.data);
            Array err = sample.label - actualOutput;
            auto backpropResult = net.backprop(err);
            net.adjustWeights(learningRate);

        }
        if (j % 50 == 49) {
            cout << "epoch " << j+1
                << " loss: " << totalLoss(L2_loss, net, trainSet) << "\n";
        }
    }
    cout << "\n";
    cout << net << "\n";

    for (auto s : trainSet) {
        cout << s.data << " -> ";
        cout << labelEnc.inverse_transform(*net.output(s.data)) << "\n";
    }
    cout << "\n";

}
