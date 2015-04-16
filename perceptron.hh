#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>


#include "classifier.hh"
#include "activation.hh"


using namespace std;


using epoch_parameter = function<double(int)>;


epoch_parameter const_epoch_parameter(double eta) {
    return [eta](int) { return eta; };
}


class APerceptron {
    /// induced local field of activation potential $v_k$, page 11
    virtual double inducedLocalField(const Input &x) const = 0;
    /// neuron's output (activation function applied to the induced local field)
    virtual double output(const Input &x) const = 0;
    /// neuron's weights; weights[0] is bias
    virtual vector<double> getWeights() const = 0;
};


class Perceptron : public APerceptron,
                   public BinaryClassifier {
private:
    double bias;
    vector<double> weights;
    const ActivationFunction &activationFunction;

    void trainConverge_addSample(Input input, double output, double eta) {
         double y = this->output(input);
         double xfactor = eta * (output - y);
         bias += xfactor * 1.0;
         transform(weights.begin(), weights.end(), input.begin(),
                   weights.begin() /* output */,
                   [&xfactor](double w_i, double x_i) { return w_i + xfactor*x_i; });
    }

    void trainBatch_addBatch(LabeledSet batch, double eta) {
        for (auto sample : batch) {
            double d = sample.output[0]; // desired output
            Input x = sample.input;
            bias += eta * 1.0 * d;
            transform(x.begin(), x.end(), weights.begin(), weights.begin(),
                      [d, eta](double x_i, double w_i) { return w_i + eta * x_i * d; });
        }
    }

public:
    Perceptron(int n, const ActivationFunction &af=defaultSignum) :
        bias(0), weights(n), activationFunction(af) {}

    virtual double inducedLocalField(const Input &x) const {
        assert (x.size() == weights.size());
        return inner_product(weights.begin(), weights.end(), x.begin(), bias);
    }

    virtual double output(const Input &x) const {
        return activationFunction(inducedLocalField(x));
    }

    virtual bool classify(const Input &x) const {
        return output(x) > 0;
    }

    /// perceptron convergence algorithm (Table 1.1)
    void trainConverge(const LabeledSet &trainSet,
                       int epochs,
                       double eta=1.0) {
        return trainConverge(trainSet, epochs, const_epoch_parameter(eta));
    }

    /// perceptron convergence algorithm (Table 1.1)
    void trainConverge(const LabeledSet &trainSet,
                       int epochs,
                       epoch_parameter eta) {
        assert (trainSet.getOutputSize() == 1);
        for (int epoch=0; epoch < epochs; ++epoch) {
            double etaval = eta(epoch);
            for (auto sample : trainSet) {
                trainConverge_addSample(sample.input, sample.output[0], etaval);
            }
        }
    }

    /// batch-training algorithm (Sec 1.6, Eq. 1.42)
    void trainBatch(const LabeledSet &trainSet,
                    int epochs,
                    double eta=1.0) {
        return trainBatch(trainSet, epochs, const_epoch_parameter(eta));
    }

    /// batch-training algorithm (Sec 1.6, Eq. 1.42)
    void trainBatch(const LabeledSet &trainSet,
                    int epochs,
                    epoch_parameter eta) {
        assert (trainSet.getOutputSize() == 1);
        assert (trainSet.getInputSize() == weights.size());
        LabeledPairPredicate isMisclassified = [this](const Input& in, const Output& out) {
            return (this->output(in))*out[0] <= 0;
        };
        // \nabla J(w) = \sum_{\vec{x}(n) \in H} ( - \vec{x}(n) d(n) )      (1.40)
        // w(n+1) = w(n) - eta(n) \nabla J(w)                               (1.42)
        for (int epoch=0; epoch < epochs; ++epoch) {
            double etaval = eta(epoch);
            // a new batch
            LabeledSet misclassifiedSet = trainSet.filter(isMisclassified);
            // sum cost gradient over the entire bactch
            trainBatch_addBatch(misclassifiedSet, etaval);
        }
    }

    virtual vector<double> getWeights() const {
        vector<double> biasAndWeights(weights);
        biasAndWeights.insert(biasAndWeights.begin(), bias);
        return biasAndWeights;
    }

    string fmt() {
        ostringstream ss;
        for (auto it : getWeights()) {
            ss << " " << it;
        }
        return ss.str();
    }

};


/**
 A basic perceptron, without built-in training facilities, with
 reasonable defaults to be used within `PerceptronsLayers`.
 */
class BasicPerceptron : public APerceptron {
private:
    vector<double> weights;  // weights[0] is bias
    const ActivationFunction& activationFunction;

public:
    BasicPerceptron(int n, const ActivationFunction &af=defaultTanh) :
        weights(n + 1), activationFunction(af) { }

    virtual double inducedLocalField(const Input &x) const {
        double bias = weights[0];
        auto weights_2nd = next(weights.begin());
        return inner_product(weights_2nd, weights.end(), x.begin(), bias);
    }

    virtual double output(const Input &x) const {
        return activationFunction(inducedLocalField(x));
    }

    virtual vector<double> getWeights() const {
        return weights;
    }
};

/// A fully connected layer of perceptrons.
class PerceptronsLayer {
private:
    int nInputs;
    int nNeurons;
    vector<BasicPerceptron> neurons;

public:
    PerceptronsLayer(int nInputs, int nOutputs,
                     const ActivationFunction &af=defaultTanh) :
        nInputs(nInputs),
        nNeurons(nOutputs),
        neurons(nOutputs, BasicPerceptron(nInputs, af)) {}

    vector<vector<double>> getWeightMatrix() const {
        vector<vector<double>> weightMatrix(0);
        for (auto n : neurons) {
            weightMatrix.push_back(n.getWeights());
        }
        return weightMatrix;
    }

    // Pages 132-133.
    Output forwardPass(const Input& xs) {
        Output output(nNeurons);
        for (int i=0; i < nNeurons; ++i) {
            output[i] = neurons[i].output(xs);
        }
        return output;
    }
};


ostream& operator<<(ostream& out, PerceptronsLayer& layer) {
    auto W = layer.getWeightMatrix();
    for (auto w_i : W) {
        for (auto w_ij : w_i) {
            out << w_ij << " ";
        }
        out << "\n";
    }
    return out;
}


ostream& operator<<(ostream& out, Output& xs) {
    int n = xs.size();
    out << "[ ";
    for (int i=0; i < n - 1; ++i) {
        out << xs[i] << ", ";
    }
    out << xs[n - 1] << " ]";
    return out;
}

#endif /* PERCEPTRON_H */
