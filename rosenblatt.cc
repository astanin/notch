// Chapter 1. Rosenblatt's Perceptron.
// Section 1.6. The Batch Perceptron Algorithm.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>


#include "dataset.hh"


using namespace std;


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class Perceptron {
    double bias;
    vector<double> weights;

    void trainPCA_addSample(vector<double> input, double output, double eta) {
         double y = response(input);
         double xfactor = eta * (output - y);
         transform(weights.begin(), weights.end(), input.begin(),
                   weights.begin() /* output */,
                   [&xfactor](double w_i, double x_i) { return w_i + xfactor*x_i; });
    }

    public:
    Perceptron(int n) : bias(0), weights(n) {}

    double inducedLocalField(const vector<double> &x) const {
        assert (x.size() == weights.size());
        return inner_product(weights.begin(), weights.end(), x.begin(), bias);
    }

    double response(const vector<double> &x) const {
        return sign(inducedLocalField(x));
    }

    /// use perceptron convergence algorithm (Table 1.1)
    void trainPCA(const LabeledSet &ts, double eta=1.0) {
        assert (ts.outputSize == 1);
        for (int i=0; i < ts.nSamples; ++i) {
            trainPCA_addSample(ts.inputs[i], ts.outputs[i][0], eta);
        }
    }

    string fmt() {
        ostringstream ss;
        ss << bias;
        for (auto it : weights) {
            ss << " " << it;
        }
        return ss.str();
    }

};


int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: rosenblatt train.data test.data\n";
        exit(-1);
    }
    ifstream trainIn(argv[1]);
    ifstream testIn(argv[2]);
    LabeledSet trainset = LabeledSet(trainIn);
    LabeledSet testset = LabeledSet(testIn);

    Perceptron p(trainset.inputSize);
    cout << "Initial weights: " << p.fmt() << "\n";
    p.trainPCA(trainset);
    cout << "Final weights: " << p.fmt() << "\n";

    int nErrors = 0;
    for (int i=0; i<testset.nSamples; ++i) {
        auto x = testset.inputs[i];
        double y_true = testset.outputs[i][0];
        double y_predicted = p.response(x);
        if (y_true * y_predicted < 0) {
            nErrors++;
            for (auto x_j : x) {
                cout << x_j << " ";
            }
            cout << "=> " << y_predicted;
            cout << " ERROR\n";
        }
    }
    cout << "Accuracy: " << (100.0*(1-nErrors*1.0/testset.nSamples)) << "%\n";
}

