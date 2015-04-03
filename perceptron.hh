#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>


#include "classifier.hh"


using namespace std;


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class Perceptron : public BinaryClassifier<double> {
    double bias;
    vector<double> weights;

    void trainConverge_addSample(Input input, double output, double eta) {
         double y = response(input);
         double xfactor = eta * (output - y);
         transform(weights.begin(), weights.end(), input.begin(),
                   weights.begin() /* output */,
                   [&xfactor](double w_i, double x_i) { return w_i + xfactor*x_i; });
    }

    public:
    Perceptron(int n) : bias(0), weights(n) {}

    double inducedLocalField(const Input &x) const {
        assert (x.size() == weights.size());
        return inner_product(weights.begin(), weights.end(), x.begin(), bias);
    }

    virtual double response(const Input &x) const {
        return sign(inducedLocalField(x));
    }

    /// use perceptron convergence algorithm (Table 1.1)
    void trainConverge(const LabeledSet &trainSet, double eta=1.0) {
        assert (trainSet.getOutputSize() == 1);
        for (auto sample : trainSet) {
            trainConverge_addSample(sample.input, sample.output[0], eta);
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

#endif /* PERCEPTRON_H */
