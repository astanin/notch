#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>
#include <cmath>      // sqrt
#include <tuple>


#include "randomgen.hh"
#include "classifier.hh"
#include "activation.hh"


using namespace std;


using epoch_parameter = function<double(int)>;


epoch_parameter const_epoch_parameter(double eta) {
    return [eta](int) { return eta; };
}


using Weights = vector<double>;


class APerceptron {
    /// induced local field of activation potential $v_k$, page 11
    virtual double inducedLocalField(const Input &x) = 0;
    /// neuron's output (activation function applied to the induced local field)
    virtual double output(const Input &x) = 0;
    /// neuron's weights; weights[0] is bias
    virtual Weights getWeights() const = 0;
};


class Perceptron : public APerceptron, public BinaryClassifier {
private:
    double bias;
    vector<double> weights;
    const ActivationFunction &activationFunction;

    void trainConverge_addSample(Input input, double output, double eta) {
        double y = this->output(input);
        double xfactor = eta * (output - y);
        bias += xfactor * 1.0;
        transform(
            weights.begin(), weights.end(), input.begin(),
            weights.begin() /* output */,
            [&xfactor](double w_i, double x_i) { return w_i + xfactor * x_i; });
    }

    void trainBatch_addBatch(LabeledSet batch, double eta) {
        for (auto sample : batch) {
            double d = sample.output[0]; // desired output
            Input x = sample.input;
            bias += eta * 1.0 * d;
            transform(x.begin(), x.end(), weights.begin(), weights.begin(),
                      [d, eta](double x_i, double w_i) {
                          return w_i + eta * x_i * d;
                      });
        }
    }

public:
    Perceptron(int n, const ActivationFunction &af = defaultSignum)
        : bias(0), weights(n), activationFunction(af) {}

    virtual double inducedLocalField(const Input &x) {
        assert(x.size() == weights.size());
        return inner_product(weights.begin(), weights.end(), x.begin(), bias);
    }

    virtual double output(const Input &x) {
        return activationFunction(inducedLocalField(x));
    }

    virtual bool classify(const Input &x) { return output(x) > 0; }

    /// perceptron convergence algorithm (Table 1.1)
    void trainConverge(const LabeledSet &trainSet, int epochs,
                       double eta = 1.0) {
        return trainConverge(trainSet, epochs, const_epoch_parameter(eta));
    }

    /// perceptron convergence algorithm (Table 1.1)
    void trainConverge(const LabeledSet &trainSet, int epochs,
                       epoch_parameter eta) {
        assert(trainSet.getOutputSize() == 1);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double etaval = eta(epoch);
            for (auto sample : trainSet) {
                trainConverge_addSample(sample.input, sample.output[0], etaval);
            }
        }
    }

    /// batch-training algorithm (Sec 1.6, Eq. 1.42)
    void trainBatch(const LabeledSet &trainSet, int epochs, double eta = 1.0) {
        return trainBatch(trainSet, epochs, const_epoch_parameter(eta));
    }

    /// batch-training algorithm (Sec 1.6, Eq. 1.42)
    void trainBatch(const LabeledSet &trainSet, int epochs,
                    epoch_parameter eta) {
        assert(trainSet.getOutputSize() == 1);
        assert(trainSet.getInputSize() == weights.size());
        LabeledPairPredicate isMisclassified =
            [this](const Input &in, const Output &out) {
                return (this->output(in)) * out[0] <= 0;
            };
        // \nabla J(w) = \sum_{\vec{x}(n) \in H} ( - \vec{x}(n) d(n) ) (1.40)
        // w(n+1) = w(n) - eta(n) \nabla J(w) (1.42)
        for (int epoch = 0; epoch < epochs; ++epoch) {
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



ostream &operator<<(ostream &out, const vector<double> &xs);


/**
 A basic perceptron, without built-in training facilities, with
 reasonable defaults to be used within `PerceptronsLayers`.
 */
class BasicPerceptron : public APerceptron {
private:
    Weights weights; // weights[0] is bias
    const ActivationFunction &activationFunction;

    // remember the last input and internal parameters to use them
    // again in the back-propagation step
    double lastInducedLocalField;  // v_j = \sum w_i y_i
    double lastActivationValue;    // y_j = \phi (v_j)
    double lastActivationGradient; // y_j = \phi^\prime (v_j)
    double lastLocalGradient;      // delta_j = \phi^\prime(v_j) e_j

public:
    BasicPerceptron(int n, const ActivationFunction &af = defaultTanh)
        : weights(n + 1), activationFunction(af) {}

    // one-sided Xavier initialization
    // see http://andyljones.tumblr.com/post/110998971763/
    void init(unique_ptr<rng_type> &rng) {
        int n_in = weights.size()-1;
        double sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
        uniform_real_distribution<double> nd(-sigma, sigma);
        generate(weights.begin(), weights.end(), [&nd, &rng] {
                    double w = nd(*rng.get());
                    return w;
                 });
    }

    virtual double inducedLocalField(const Input &x) {
        double bias = weights[0];
        auto weights_2nd = next(weights.begin());
        return inner_product(weights_2nd, weights.end(), x.begin(), bias);
    }

    virtual double output(const Input &x) {
        double v = inducedLocalField(x);
        lastInducedLocalField = v;
        lastActivationValue = activationFunction(v);
        lastActivationGradient = activationFunction.derivative(v);
        return lastActivationValue;
    }

    virtual Weights getWeights() const {
        return weights;
    }

    void adjustWeights(Weights deltaW) {
        assert(deltaW.size() == weights.size());
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += deltaW[i];
        }
    }

    struct BPResult {
        Weights weightCorrection;
        double localGradient;
    };

    // Page 134. Equation (4.27) defines weight correction
    //
    // $$ \Delta w_{ji} (n) =
    //    \eta
    //      \times
    //    \delta_j (n)
    //      \times
    //    y_{i} (n) $$
    //
    // where $w_{ji}$ is the synaptic weight connecting neuron $i$ to neuron $j$,
    // $\eta$ is learning rate, $delta_j (n)$ is the local [error] gradient,
    // $y_{i}$ is the input signal of the neuron $i$, $n$ is the epoch number
    //
    // The local gradient is the product of the activation function derivative
    // and the error signal.
    //
    // Return a vector of weight corrections and the local gradient value.
    //
    // The method should be called _after_ `forwardPass`
    BPResult backwardPass(const Input &inputs, double errorSignal,
                          double learningRate) {
        assert(inputs.size() + 1 == weights.size());
        size_t nInputs = weights.size();
        double localGradient = lastActivationGradient * errorSignal;
        double multiplier = learningRate * localGradient;
        Weights delta_W(nInputs, multiplier);
        for (size_t i = 0; i < inputs.size(); ++i) {
            delta_W[i + 1] *= inputs[i];
        }
        lastLocalGradient = localGradient;
        return BPResult{delta_W, localGradient};
    }
};


/// A fully connected layer of perceptrons.
class PerceptronsLayer {
private:
    unsigned int nInputs;
    unsigned int nNeurons;
    vector<BasicPerceptron> neurons;
    Output lastOutput;

public:
    PerceptronsLayer(unsigned int nInputs, unsigned int nOutputs,
                     const ActivationFunction &af = defaultTanh)
        : nInputs(nInputs), nNeurons(nOutputs),
          neurons(nOutputs, BasicPerceptron(nInputs, af)),
          lastOutput(nOutputs) {}

    void init(unique_ptr<rng_type> &rng) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].init(rng);
        }
    }

    vector<Weights> getWeightMatrix() const {
        vector<Weights> weightMatrix(0);
        for (auto n : neurons) {
            Weights ws = n.getWeights();
            weightMatrix.push_back(ws);
        }
        return weightMatrix;
    }

    void adjustWeights(vector<Weights> weightDeltas) {
        assert(nNeurons == weightDeltas.size());
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].adjustWeights(weightDeltas[i]);
        }
    }

    // Pages 132-133.
    // Return a vector of outputs.
    Output forwardPass(const Input &inputs) {
        for (auto i = 0u; i < nNeurons; ++i) {
            lastOutput[i] = neurons[i].output(inputs);
        }
        return lastOutput;
    }

    struct BPResult {
        Output propagatedErrorSignal;
        vector<Weights> weightCorrections;
    };

    // Page 134. Update synaptic weights.
    //
    // Return a vector of back-propagated error signal
    //
    // $$ e_j = \sum_k \delta_k w_{kj} $$
    //
    // where $e_j$ is an error propagated from all downstream neurons to the
    // neuron $j$, $\delta_k$ is the local gradient of the downstream neurons
    // $k$, $w_{kj}$ is the synaptic weight of the $j$-th input of the
    // downstream neuron $k$.
    //
    // The method should be called _after_ `forwardPass`
    BPResult backwardPass(const Input &inputs, const Output &errorSignals,
                          double learningRate) {
        assert(errorSignals.size() == neurons.size());
        auto eta = learningRate;
        vector<Weights> weightDeltas(0);
        Weights propagatedErrorSignals(nInputs, 0.0);
        for (auto k = 0u; k < nNeurons; ++k) {
            auto error_k = errorSignals[k];
            auto r = neurons[k].backwardPass(inputs, error_k, eta);
            Weights delta_Wk = r.weightCorrection;
            double delta_k = r.localGradient;
            Weights Wk = neurons[k].getWeights();
            for (auto j = 0u; j < nInputs; ++j) {
                propagatedErrorSignals[j] += delta_k * Wk[j + 1];
            }
            weightDeltas.push_back(delta_Wk);
        }
        return BPResult{propagatedErrorSignals, weightDeltas};
    }
};


ostream &operator<<(ostream &out, PerceptronsLayer &layer) {
    auto W = layer.getWeightMatrix();
    for (size_t j = 0; j < W.size(); ++j) {
        if (j == 0) {
            out << "[";
        } else {
            out << " ";
        }
        out << W[j];
        if (j >= W.size() - 1) {
            out << "]";
        } else {
            out << ",\n";
        }
    }
    return out;
}


ostream &operator<<(ostream &out, const vector<double> &xs) {
    int n = xs.size();
    out << "[ ";
    for (int i = 0; i < n - 1; ++i) {
        out << xs[i] << ", ";
    }
    out << xs[n - 1] << " ]";
    return out;
}

#endif /* PERCEPTRON_H */
