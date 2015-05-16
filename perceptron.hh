#ifndef PERCEPTRON_H
#define PERCEPTRON_H


// remove these includes:
#include <assert.h>
#include <iostream>   // cout

#include <algorithm>  // generate
#include <array>      // array
#include <cmath>      // sqrt
#include <functional> // ref
#include <initializer_list>
#include <iomanip>    // setw, setprecision
#include <iterator>   // begin, end
#include <memory>     // unique_ptr
#include <numeric>    // inner_product
#include <ostream>    // ostream
#include <random>
#include <valarray>
#include <vector>


#include "dataset.hh"
#include "activation.hh"

/**
 * Framework
 * =========
 *
 **/

/**
 * Random initialization
 * ---------------------
 **/
using RNG = std::mt19937;

/// Create and seed a new random number generator.
std::unique_ptr<RNG> newRNG() {
    std::random_device rd;
    std::array<uint32_t, std::mt19937::state_size> seed_data;
    std::generate(seed_data.begin(), seed_data.end(), ref(rd));
    std::seed_seq sseq(std::begin(seed_data), std::end(seed_data));
    std::unique_ptr<RNG> rng(new RNG());
    rng->seed(sseq);
    return rng;
}


/**
 * Neurons and Neural Networks
 * ===========================
 *
 **/

/// Synaptic weights
using Weights = std::valarray<double>;


// TODO: use iterators rather than const Input&
class ANeuron {
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


class LinearPerceptron : public ANeuron {
private:
    Weights weights;
    const ActivationFunction &activationFunction;

public:
    LinearPerceptron(int n, const ActivationFunction &af = linearActivation)
        : weights(n + 1), activationFunction(af) {}

    virtual double inducedLocalField(const Input &x) {
        assert(x.size() + 1 == weights.size());
        double bias = weights[0];
        auto begin_weights = std::next(std::begin(weights));
        return std::inner_product(begin_weights, std::end(weights), std::begin(x), bias);
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


std::ostream &operator<<(std::ostream &out, const ANeuron &neuron) {
    auto ws = neuron.getWeights();
    for (auto it = std::begin(ws); it != std::end(ws); ++it) {
        if (std::next(it) != std::end(ws)) {
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
void trainConverge_addSample(ANeuron &p, Input input, double output, double eta) {
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

void trainConverge(ANeuron &p, const LabeledDataset &trainSet,
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
void trainBatch_addBatch(ANeuron &p, LabeledDataset batch, double eta) {
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

void trainBatch(ANeuron &p, const LabeledDataset &trainSet, int epochs, double eta) {
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
class BidirectionalNeuron : public ANeuron {
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
    // TODO: move algorithm outside of the class
    void init(std::unique_ptr<RNG> &rng) {
        int n_in = nInputs;
        double sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
        std::uniform_real_distribution<double> nd(-sigma, sigma);
        std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
                    double w = nd(*rng.get());
                    return w;
                 });
    }

    virtual double inducedLocalField(const Input &x) {
        auto bias = weights[0];
        auto begin_weights = std::next(std::begin(weights));
        return std::inner_product(begin_weights, std::end(weights), std::begin(x), bias);
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

    friend std::ostream &operator<<(std::ostream &out, const BidirectionalNeuron &neuron);
};


/// A fully connected layer of a multilayer perceptron.
class FullyConnectedLayer {
private:
    unsigned int nInputs;
    unsigned int nNeurons;
    std::vector<BidirectionalNeuron> neurons;
    Output lastOutput;

public:
    FullyConnectedLayer(unsigned int nInputs = 0, unsigned int nOutputs = 0,
                     const ActivationFunction &af = defaultTanh)
        : nInputs(nInputs), nNeurons(nOutputs),
          neurons(nOutputs, BidirectionalNeuron(nInputs, af)),
          lastOutput(nOutputs) {}

    void init(std::unique_ptr<RNG> &rng) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].init(rng);
        }
    }

    void adjustWeights(std::vector<Weights> weightDeltas) {
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
        std::vector<Weights> weightCorrections;
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
        std::vector<Weights> weightDeltas(0);
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

    friend std::ostream &operator<<(std::ostream &out, const FullyConnectedLayer &net);
};


/// Multiple fully-connected layers stacked one upon another.
class MultilayerPerceptron {
private:
    std::vector<FullyConnectedLayer> layers;
    std::vector<Input> layersInputs;

public:
    MultilayerPerceptron(std::initializer_list<unsigned int> shape,
                         const ActivationFunction &af = defaultTanh)
        : layers(0), layersInputs(0) {
        auto pIn = shape.begin();
        auto pOut = std::next(pIn);
        for (; pOut != shape.end(); ++pIn, ++pOut) {
            FullyConnectedLayer layer(*pIn, *pOut, af);
            layers.push_back(layer);
        }
    }

    void init(std::unique_ptr<RNG> &rng) {
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

    friend std::ostream &operator<<(std::ostream &out, const MultilayerPerceptron &net);
};


std::ostream &operator<<(std::ostream &out, const BidirectionalNeuron &neuron) {
    auto weights = neuron.getWeights();
    for (auto w : weights) {
        out << std::setw(9) << std::setprecision(5) << w << " ";
    }
    out << neuron.activationFunction;
    return out;
}


std::ostream &operator<<(std::ostream &out, const FullyConnectedLayer &layer) {
    for (BidirectionalNeuron neuron : layer.neurons) {
        out << "  " << neuron << "\n";
    }
    return out;
}


std::ostream &operator<<(std::ostream &out, const MultilayerPerceptron &net) {
    int layerN = 1;
    for (FullyConnectedLayer l : net.layers) {
        out << "LAYER " << layerN << ":\n";
        out << l;
        layerN++;
    }
    return out;
}


#endif /* PERCEPTRON_H */
