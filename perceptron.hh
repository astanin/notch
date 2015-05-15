#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <sstream>
#include <vector>
#include <valarray>
#include <numeric>    // inner_product
#include <algorithm>  // transform
#include <functional> // plus, minus
#include <assert.h>
#include <cmath>      // sqrt
#include <initializer_list>
#include <iostream>   // cout
#include <iomanip>    // setw, setprecision
#include <iterator>   // begin, end


#include "randomgen.hh"
#include "dataset.hh"
#include "activation.hh"


using namespace std;


using Weights = valarray<double>;


class APerceptron {
public:
    /// induced local field of activation potential $v_k$, page 11, eq (4)
    ///
    /// $$ v_k = \sum_{j = 0}^{m} w_{kj} x_j, $$
    ///
    /// where $w_{kj}$ is the weight of the $j$-th input of the neuron $k$,
    /// and $x_j$ is the $j$-th input.
    virtual double inducedLocalField(const Input &x) = 0;
    /// neuron's output, page 12, eq (5)
    ///
    /// $$ y_k = \varphi (v_k) $$
    ///
    /// @return the value activation function applied to the induced local field
    virtual double output(const Input &x) = 0;
    /// get neuron's weights; weights[0] ($w_{k0}$) is bias
    virtual Weights getWeights() const = 0;
    /// add weight correction to the neuron's weights
    virtual Weights adjustWeights(const Weights weightCorrections) = 0;
};


class LinearPerceptron : public APerceptron {
private:
    Weights weights;
    const ActivationFunction &activationFunction;

public:
    LinearPerceptron(int n, const ActivationFunction &af = linearActivation)
        : weights(n + 1), activationFunction(af) {}

    virtual double inducedLocalField(const Input &x) {
        assert(x.size() + 1 == weights.size());
        double bias = weights[0];
        auto begin_weights = next(begin(weights));
        return inner_product(begin_weights, end(weights), begin(x), bias);
    }

    virtual double output(const Input &x) {
        return activationFunction(inducedLocalField(x));
    }

    virtual Weights getWeights() const {
        return weights;
    }

    virtual Weights adjustWeights(const Weights weightCorrections) {
        assert(weights.size() == weightCorrections.size());
        weights += weightCorrections;
        return weights;
    }
};


ostream &operator<<(ostream &out, const APerceptron &neuron) {
    auto ws = neuron.getWeights();
    for (auto it = begin(ws); it != end(ws); ++it) {
        if (next(it) != end(ws)) {
            out << *it << " ";
        } else {
            out << *it;
        }
    }
    return out;
}

/**
 * Perceptron convergence algorithm
 * --------------------------------
 *
 * See (Table 1.1).
 *
 *  1. Initialize $\mathbf{w}(0) = \mathbf{0}$.
 *
 *  2. For $n = 1, 2, ...$:
 *
 *     *  active the perceptron with an input vector $\mathbf{x}(n)$
 *        and consider the desired response $d(n)$ (+1 or -1).
 *
 *     *. Apply signum activation function
 *
 *        $$ y(n) = sign( \mathbf{w}^T(n) \mathbf{x}(n) ) $$
 *
 *     *. Update the weight vector as follows:
 *
 *        $$ \mathbf{w}(n+1) = \mathbf{w}() +
 *                             \eta (d(n) - y(n)) \mathbf{x}(n), $$
 *
 *        where $\eta$ is learning rate.
 **/
void trainConverge_addSample(APerceptron &p, Input input, double output, double eta) {
    double y = sign(p.output(input));
    double xfactor = eta * (output - y);
    Weights weights = p.getWeights();
    // initialize all corrections as if they're multiplied by xfactor
    Weights deltaW(xfactor, weights.size());
    // deltaW[0] *= 1.0; // bias, no-op
    for (size_t i = 0; i < input.size(); ++i) {
        deltaW[i+1] *= input[i];
    }
    p.adjustWeights(deltaW);
}

void trainConverge(APerceptron &p, const LabeledDataset &trainSet,
                   int epochs, double eta) {
    assert(trainSet.outputDim() == 1);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (auto sample : trainSet) {
            trainConverge_addSample(p, sample.data, sample.label[0], eta);
        }
    }
}


/** The batch perceptron training algorithm
 *  ---------------------------------------
 *
 * (Sec 1.6, Eq. 1.42)
 *
 * Consider the _perceptron cost function_
 *
 * $$ J(\mathbf{w}) = \sum_{\mathbf{x}(n) \in \Xi}
 *                    ( - \mathbf{w}^T \mathbf{x}(n) d(n) ), $$
 *
 * where $\Xi$ is the set of the misclassified samples $\mathbf{x}.
 * Differentiating the cost $J(\mathbf{w})$ with respect to $\mathbf{w}$
 * gives the _gradient vector_
 *
 * $$ \nabla J(\mathbf{w}) = \sum_{\mathbf{x}(n) \in \Xi}
 *                           ( - \mathbf{x}(n) d(n) ). $$
 *
 * The adjustment to the weight vector is applied in a direction
 * opposite to the gradient vector:
 *
 * $$ \mathbf{w}(n+1) = \mathbf{w}(n) - \eta(n) \nabla J(\mathbf{w})
 *       = \mathbf{w}(n) + \eta(n) \sum_{\mathbf{x}(n) \in \Xi}
 *                                      ( - \mathbf{x}(n) d(n) ). $$
 **/
void trainBatch_addBatch(APerceptron &p, LabeledDataset batch, double eta) {
    for (auto sample : batch) {
        double desired = sample.label[0]; // desired output
        Input input = sample.data;
        Weights weights = p.getWeights();
        // initialize all corrections as if multiplied by eta*desired
        Weights deltaW(eta * desired, weights.size());
        // deltaW[0] *= 1.0; // bias, no-op
        for (size_t i = 0; i < input.size(); ++i) {
            deltaW[i+1] *= input[i];
        }
        p.adjustWeights(deltaW);
    }
}

void trainBatch(APerceptron &p, const LabeledDataset &trainSet, int epochs, double eta) {
    assert(trainSet.outputDim() == 1);
    assert(trainSet.inputDim() + 1 == p.getWeights().size());
    // \nabla J(w) = \sum_{\vec{x}(n) \in H} ( - \vec{x}(n) d(n) ) (1.40)
    // w(n+1) = w(n) - eta(n) \nabla J(w) (1.42)
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // a new batch
        LabeledDataset misclassifiedSet;
        for (auto sample : trainSet) {
           if (p.output(sample.data) * sample.label[0] <= 0) {
              misclassifiedSet.append(sample);
           }
        }
        // sum cost gradient over the entire bactch
        trainBatch_addBatch(p, misclassifiedSet, eta);
    }
}


/**
 * An artificial neuron with back-propagation capability.
 */
class BidirectionalNeuron : public APerceptron {
private:
    int nInputs;
    Weights weights; // weights[0] is bias
    const ActivationFunction &activationFunction;

    // remember the latest internal parameters to use them
    // again in the back-propagation step
    double lastInducedLocalField;  // v_j = \sum w_i y_i
    double lastActivationValue;    // y_j = \phi (v_j)
    double lastActivationGradient; // y_j = \phi^\prime (v_j)
    double lastLocalGradient;      // delta_j = \phi^\prime(v_j) e_j

public:
    BidirectionalNeuron(int n, const ActivationFunction &af = defaultTanh)
        : nInputs(n), weights(n + 1), activationFunction(af) {}

    // one-sided Xavier initialization
    // see http://andyljones.tumblr.com/post/110998971763/
    void init(unique_ptr<rng_type> &rng) {
        int n_in = nInputs;
        double sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
        uniform_real_distribution<double> nd(-sigma, sigma);
        generate(begin(weights), end(weights), [&nd, &rng] {
                    double w = nd(*rng.get());
                    return w;
                 });
    }

    virtual double inducedLocalField(const Input &x) {
        auto bias = weights[0];
        auto begin_weights = next(begin(weights));
        return inner_product(begin_weights, end(weights), begin(x), bias);
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

    virtual Weights adjustWeights(Weights weightCorrections) {
        assert(weights.size() == weightCorrections.size());
        weights += weightCorrections;
        return weights;
    }

    struct BackOutput {
        double localGradient;
        Weights weightCorrections;
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
    BackOutput backwardPass(const Input &inputs, double errorSignal,
                            double learningRate) {
        double localGradient = lastActivationGradient * errorSignal;
        double multiplier = learningRate * localGradient;
        Weights delta_W(multiplier, weights.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            delta_W[i + 1] *= inputs[i];
        }
        lastLocalGradient = localGradient;
        BackOutput ret{localGradient, delta_W};
        return ret;
    }

    friend ostream &operator<<(ostream &out, const BidirectionalNeuron &neuron);
};


/// A fully connected layer of a multilayer perceptron.
class FullyConnectedLayer {
private:
    unsigned int nInputs;
    unsigned int nNeurons;
    vector<BidirectionalNeuron> neurons;
    Output lastOutput;

public:
    FullyConnectedLayer(unsigned int nInputs = 0, unsigned int nOutputs = 0,
                     const ActivationFunction &af = defaultTanh)
        : nInputs(nInputs), nNeurons(nOutputs),
          neurons(nOutputs, BidirectionalNeuron(nInputs, af)),
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

    struct BackOutput {
        Input propagatedErrorSignals;
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
    BackOutput backwardPass(const Input &inputs, const Output &errorSignals,
                                 double learningRate) {
        assert(errorSignals.size() == neurons.size());
        auto eta = learningRate;
        vector<Weights> weightDeltas(0);
        Weights propagatedErrorSignals(0.0, nInputs);
        for (auto k = 0u; k < nNeurons; ++k) {
            auto error_k = errorSignals[k];
            auto r = neurons[k].backwardPass(inputs, error_k, eta);
            Weights delta_Wk = r.weightCorrections;
            double delta_k = r.localGradient;
            Weights Wk = neurons[k].getWeights();
            for (auto j = 0u; j < nInputs; ++j) {
                propagatedErrorSignals[j] += delta_k * Wk[j + 1];
            }
            weightDeltas.push_back(delta_Wk);
        }
        return BackOutput{propagatedErrorSignals, weightDeltas};
    }

    friend ostream &operator<<(ostream &out, const FullyConnectedLayer &net);
};


/// Multiple fully-connected layers stacked one upon another.
class MultilayerPerceptron {
private:
    vector<FullyConnectedLayer> layers;
    vector<Input> layersInputs;

public:
    MultilayerPerceptron(initializer_list<unsigned int> shape,
                       const ActivationFunction &af = defaultTanh)
        : layers(0), layersInputs(0) {
        auto pIn = shape.begin();
        auto pOut = next(pIn);
        for (; pOut != shape.end(); ++pIn, ++pOut) {
            FullyConnectedLayer layer(*pIn, *pOut, af);
            layers.push_back(layer);
        }
    }

    void init(unique_ptr<rng_type> &rng) {
        for (auto i = 0u; i < layers.size(); ++i) {
            layers[i].init(rng);
        }
    }

    Output forwardPass(const Input &inputs) {
        // TODO: avoid allocationg new Inputs on every pass
        layersInputs.clear();
        layersInputs.push_back(inputs);
        for (auto i = 0u; i < layers.size(); ++i) {
            auto in = layersInputs[i];
            auto out = layers[i].forwardPass(in);
            layersInputs.push_back(out);
        }
        return layersInputs[layers.size()];
    }

    FullyConnectedLayer::BackOutput
    backwardPass(const Output &errorSignals, double learningRate) {
        Output err(errorSignals);
        FullyConnectedLayer::BackOutput r;

        for (int i = layers.size()-1; i >= 0; --i) {
            auto layerIn = layersInputs[i];
            r = layers[i].backwardPass(layerIn, err, learningRate);
            layers[i].adjustWeights(r.weightCorrections);
            err = r.propagatedErrorSignals;
        }
        return r;
    }

    friend ostream &operator<<(ostream &out, const MultilayerPerceptron &net);
};


ostream &operator<<(ostream &out, const BidirectionalNeuron &neuron) {
    auto weights = neuron.getWeights();
    for (auto w : weights) {
        out << setw(9) << setprecision(5) << w << " ";
    }
    out << neuron.activationFunction;
    return out;
}


ostream &operator<<(ostream &out, const FullyConnectedLayer &layer) {
    for (BidirectionalNeuron neuron : layer.neurons) {
        out << "  " << neuron << "\n";
    }
    return out;
}


ostream &operator<<(ostream &out, const MultilayerPerceptron &net) {
    int layerN = 1;
    for (FullyConnectedLayer l : net.layers) {
        out << "LAYER " << layerN << ":\n";
        out << l;
        layerN++;
    }
    return out;
}


#endif /* PERCEPTRON_H */
