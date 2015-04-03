#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>


using namespace std;


double sign(double a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class Perceptron {
    double bias;
    vector<double> weights;

    void trainConverge_addSample(vector<double> input, double output, double eta) {
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
    void trainConverge(const LabeledSet &ts, double eta=1.0) {
        assert (ts.outputSize == 1);
        for (int i=0; i < ts.nSamples; ++i) {
            trainConverge_addSample(ts.inputs[i], ts.outputs[i][0], eta);
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
