#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>
#include <cmath>      // sqrt
#include <initializer_list>
#include <iostream>   // cout
#include <iterator>   // begin, end


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


// TODO: rename to LinearPerceptron
class StandalonePerceptron : public APerceptron, public BinaryClassifier {
private:
    // TODO: use weights[0] as bias
    double bias;
    Weights weights;
    const ActivationFunction &activationFunction;

    // TODO: implement via adjustWeights()
    void trainConverge_addSample(Input input, double output, double eta) {
        double y = this->output(input);
        double xfactor = eta * (output - y);
        bias += xfactor * 1.0;
        transform(
            weights.begin(), weights.end(), begin(input),
            weights.begin() /* output */,
            [&xfactor](double w_i, double x_i) { return w_i + xfactor * x_i; });
    }

    void trainBatch_addBatch(LabeledSet batch, double eta) {
        for (auto sample : batch) {
            double d = sample.label[0]; // desired output
            Input x = sample.data;
            bias += eta * 1.0 * d;
            transform(begin(x), end(x), weights.begin(), weights.begin(),
                      [d, eta](double x_i, double w_i) {
                          return w_i + eta * x_i * d;
                      });
        }
    }

public:
    StandalonePerceptron(int n, const ActivationFunction &af = defaultSignum)
        : bias(0), weights(n), activationFunction(af) {}

    virtual double inducedLocalField(const Input &x) {
        assert(x.size() == weights.size());
        return inner_product(weights.begin(), weights.end(), begin(x), bias);
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
        assert(trainSet.outputDim() == 1);
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double etaval = eta(epoch);
            for (auto sample : trainSet) {
                trainConverge_addSample(sample.data, sample.label[0], etaval);
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
        assert(trainSet.outputDim() == 1);
        assert(trainSet.inputDim() == weights.size());
        // \nabla J(w) = \sum_{\vec{x}(n) \in H} ( - \vec{x}(n) d(n) ) (1.40)
        // w(n+1) = w(n) - eta(n) \nabla J(w) (1.42)
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double etaval = eta(epoch);
            // a new batch
            LabeledSet misclassifiedSet;
            for (auto sample : trainSet) {
               if (output(sample.data) * sample.label[0] <= 0) {
                  misclassifiedSet.append(sample);
               }
            }
            // sum cost gradient over the entire bactch
            trainBatch_addBatch(misclassifiedSet, etaval);
        }
    }

    virtual vector<double> getWeights() const {
        vector<double> biasAndWeights(weights);
        biasAndWeights.insert(biasAndWeights.begin(), bias);
        return biasAndWeights;
    }

    // TODO: move to non-member function
    string fmt() {
        ostringstream ss;
        for (auto it : getWeights()) {
            ss << " " << it;
        }
        return ss.str();
    }
};



ostream &operator<<(ostream &out, const vector<double> &xs);

#if 0

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
    PerceptronsLayer(unsigned int nInputs = 0, unsigned int nOutputs = 0,
                     const ActivationFunction &af = defaultTanh)
        : nInputs(nInputs), nNeurons(nOutputs),
          neurons(nOutputs, BasicPerceptron(nInputs, af)),
          lastOutput(nOutputs) {}

    void init(unique_ptr<rng_type> &rng) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].init(rng);
        }
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
        Output propagatedErrorSignals;
        vector<Weights> weightCorrections;
    };

    // Page 134. Calculate back-propagated error signal and corrections to
    // synaptic weights.
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

    friend ostream &operator<<(ostream &out, const PerceptronsLayer &net);
};


/// Multiple layers of perceptrons stack one upon another.
// TODO: rename to MultilayerPerceptron
class PerceptronsNetwork {
private:
    vector<PerceptronsLayer> layers;
    vector<Input> layersInputs;

public:
    PerceptronsNetwork(initializer_list<unsigned int> shape,
                       const ActivationFunction &af = defaultTanh)
        : layers(0), layersInputs(0) {
        auto pIn = shape.begin();
        auto pOut = next(pIn);
        for (; pOut != shape.end(); ++pIn, ++pOut) {
            PerceptronsLayer layer(*pIn, *pOut, af);
            layers.push_back(layer);
        }
    }

    void init(unique_ptr<rng_type> &rng) {
        for (auto i = 0u; i < layers.size(); ++i) {
            layers[i].init(rng);
        }
    }

    Output forwardPass(const Input &inputs) {
        layersInputs.clear();
        layersInputs.push_back(inputs);
        for (auto i = 0u; i < layers.size(); ++i) {
            auto in = layersInputs[i];
            auto out = layers[i].forwardPass(in);
            layersInputs.push_back(out);
        }
        return layersInputs[layers.size()];
    }

    PerceptronsLayer::BPResult
    backwardPass(const Output &errorSignals, double learningRate) {
        Output err(errorSignals);
        PerceptronsLayer::BPResult r;

        for (int i = layers.size()-1; i >= 0; --i) {
            auto layerIn = layersInputs[i];
            r = layers[i].backwardPass(layerIn, err, learningRate);
            layers[i].adjustWeights(r.weightCorrections);
            err = r.propagatedErrorSignals;
        }
        return r;
    }

    friend ostream &operator<<(ostream &out, const PerceptronsNetwork &net);
};


ostream &operator<<(ostream &out, const PerceptronsLayer &layer) {
    size_t n = layer.neurons.size();
    for (size_t j = 0; j < n; ++j) {
        if (j == 0) {
            out << "[";
        } else {
            out << " ";
        }
        out << layer.neurons[j].getWeights();
        if (j >= n - 1) {
            out << "]";
        } else {
            out << ",\n";
        }
    }
    return out;
}


ostream &operator<<(ostream &out, const PerceptronsNetwork &net) {
    for (PerceptronsLayer l : net.layers) {
        out << l << "\n";
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

#endif

#endif /* PERCEPTRON_H */
