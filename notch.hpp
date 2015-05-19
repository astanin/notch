#ifndef NOTCH_H
#define NOTCH_H

/// notch.hpp -- main header file of the Notch neural networks library

/**

The MIT License (MIT)

Copyright (c) 2015 Sergey Astanin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

// TODO: remove these includes:
#include <assert.h>
#include <iostream>   // cout

#include <algorithm>  // generate
#include <array>      // array
#include <cmath>      // sqrt, exp
#include <functional> // ref, function<>
#include <initializer_list>
#include <iomanip>    // setw, setprecision
#include <iterator>   // begin, end
#include <memory>     // unique_ptr
#include <numeric>    // inner_product
#include <ostream>    // ostream
#include <random>
#include <string>
#include <typeinfo>   // typeid
#include <valarray>
#include <vector>


/**
 * Library Framework
 * =================
 **/

/**
 * Data types
 * ----------
 *
 * A neural network consumes a vector of numerical values, and produces a vector
 * of numerical outputs. Without too much loss of generality we may consider
 * them arrays of single-precision floating point numbers.
 *
 * We use C++ `valarray` to store network `Input` and `Output` to make code
 * more concise and expressive (valarrays implement elementwise operations and
 * slices).
 **/
using Input = std::valarray<float>;
using Output = std::valarray<float>;


/** Supervised learning requires labeled data.
 *  A label is a vector of numeric values.
 *  For classification problems it is often a vector with only one element. */
struct LabeledData {
    Input const &data;
    Output const &label;
};


/** A `LabeledDataset` consists of multiple `LabeledData` samples.
 *  `LabeledDataset`s can be used like training or testing sets.
 */
class LabeledDataset {
private:
    size_t nSamples;
    size_t inputDimension;
    size_t outputDimension;
    std::vector<Input> inputs;
    std::vector<Output> outputs;

public:

    /// An iterator type to process all labeled data samples.
    class DatasetIterator : public std::iterator<std::input_iterator_tag, LabeledData> {
    private:
        using ArrayVecIter = std::vector<Input>::const_iterator;
        ArrayVecIter in_position, in_end;
        ArrayVecIter out_position, out_end;

    public:
        DatasetIterator(ArrayVecIter in_begin,
                        ArrayVecIter in_end,
                        ArrayVecIter out_begin,
                        ArrayVecIter out_end)
            : in_position(in_begin), in_end(in_end),
              out_position(out_begin), out_end(out_end) {}

        bool operator==(const DatasetIterator &rhs) const {
            return (typeid(*this) == typeid(rhs) &&
                    in_position == rhs.in_position &&
                    out_position == rhs.out_position &&
                    in_end == rhs.in_end &&
                    out_end == rhs.out_end);
        }

        bool operator!=(const DatasetIterator &rhs) const {
            return !(*this == rhs);
        }

        LabeledData operator*() const {
            const Input &in(*in_position);
            const Output &out(*out_position);
            LabeledData lp{in, out};
            return lp;
        }

        DatasetIterator &operator++() {
            if (in_position != in_end && out_position != out_end) {
                ++in_position;
                ++out_position;
            }
            return *this;
        }

        DatasetIterator &operator++(int) { return ++(*this); }
    };

    // constructors
    LabeledDataset() : nSamples(0), inputDimension(0), outputDimension(0) {}
    LabeledDataset(std::initializer_list<LabeledData> samples)
        : nSamples(0), inputDimension(0), outputDimension(0) {
        for (LabeledData s : samples) {
            append(s);
        }
    }

    DatasetIterator begin() const {
        return DatasetIterator(
                inputs.begin(), inputs.end(),
                outputs.begin(), outputs.end());
    }

    DatasetIterator end() const {
        return DatasetIterator(
                inputs.end(), inputs.end(),
                outputs.end(), outputs.end());
    }

    size_t size() const { return nSamples; }
    size_t inputDim() const { return inputDimension; }
    size_t outputDim() const { return outputDimension; }

    LabeledDataset &append(Input &input, Output &output) {
        if (nSamples != 0) {
            assert(inputDimension == input.size());
            assert(outputDimension == output.size());
        } else {
            inputDimension = input.size();
            outputDimension = output.size();
        }
        nSamples++;
        inputs.push_back(input);
        outputs.push_back(output);
        return *this;
    }

    LabeledDataset &append(const Input &input, const Output &output) {
        if (nSamples != 0) {
            assert(inputDimension == input.size());
            assert(outputDimension == output.size());
        } else {
            inputDimension = input.size();
            outputDimension = output.size();
        }
        nSamples++;
        Input input_copy(input);
        Output output_copy(output);
        inputs.push_back(input_copy);
        outputs.push_back(output_copy);
        return *this;
    }

    LabeledDataset &append(LabeledData &sample) {
        return append(sample.data, sample.label);
    }

    LabeledDataset &append(const LabeledData &sample) {
        return append(sample.data, sample.label);
    }
};


/**
 * Neurons and Neural Networks
 * ===========================
 **/


/** Synaptic weights */
using Weights = std::valarray<float>;


/**
 * Random Weights Initialization
 * -----------------------------
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

using WeightsInitializer = std::function<void(std::unique_ptr<RNG> &, Weights &)>;

/** One-sided Xavier initialization.
 *
 * Pick weights from a zero-centered _normal_ distrubition with variance
 * $$\sigma^2 = 1/n_{in}$$, where $n_{in}$ is the number of inputs.
 *
 * See http://andyljones.tumblr.com/post/110998971763/ **/
void normalXavier(std::unique_ptr<RNG> &rng, Weights &weights) {
    size_t n_in = weights.size();
    float sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
    std::normal_distribution<float> nd(0.0, sigma);
    std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
                float w = nd(*rng.get());
                return w;
             });
}

/** Uniform one-sided Xavier initialization.
 *
 * Pick weights from a zero-centered _uniform_ distrubition with variance
 * $$\sigma^2 = 1/n_{in}$$, where $n_{in}$ is the number of inputs.
 *
 * See http://andyljones.tumblr.com/post/110998971763/ **/
void uniformXavier(std::unique_ptr<RNG> &rng, Weights &weights) {
    size_t n_in = weights.size();
    float sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
    float a = sigma * sqrt(3.0);
    std::uniform_real_distribution<float> nd(-a, a);
    std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
                float w = nd(*rng.get());
                return w;
             });
}

// TODO: use iterators rather than const Input&
/** ANeuron is an abstract neuron class. **/
class ANeuron {
public:
    /// induced local field of activation potential $v_k$, page 11, eq (4)
    ///
    /// $$ v_k = \sum_{j = 0}^{m} w_{kj} x_j, $$
    ///
    /// where $w_{kj}$ is the weight of the $j$-th input of the neuron $k$,
    /// and $x_j$ is the $j$-th input.
    virtual float inducedLocalField(const Input &x) = 0;
    /// neuron's output, page 12, eq (5)
    ///
    /// $$ y_k = \varphi (v_k) $$
    ///
    /// @return the value activation function applied to the induced local field
    virtual float output(const Input &x) = 0;
    /// get neuron's weights; weights[0] ($w_{k0}$) is bias
    virtual Weights getWeights() const = 0;
    /// add weight correction to the neuron's weights
    virtual Weights adjustWeights(const Weights &weightCorrections) = 0;
};


/**
 * Neuron Activation Functions
 * ---------------------------
 **/

float sign(float a) { return (a == 0) ? 0 : (a < 0 ? -1 : 1); }


class ActivationFunction {
public:
    virtual float operator()(float v) const = 0;
    virtual float derivative(float v) const = 0;
    virtual void print(std::ostream &out) const = 0;
};


/// phi(v) = 1/(1 + exp(-slope*v)); Chapter 4, page 135
class LogisticActivation : public ActivationFunction {
private:
    float slope = 1.0;

public:
    LogisticActivation(float slope) : slope(slope){};

    virtual float operator()(float v) const {
        return 1.0 / (1.0 + exp(-slope * v));
    }

    virtual float derivative(float v) const {
        float y = (*this)(v);
        return slope * y * (1 - y);
    }

    virtual void print(std::ostream &out) const { out << "logistic"; }
};


class SignumActivation : public ActivationFunction {
public:
    SignumActivation() {}

    virtual float operator()(float v) const { return sign(v); }

    virtual float derivative(float) const { return 0.0; }

    virtual void print(std::ostream &out) const { out << "sign"; }
};


/// phi(v) = a * tanh(b * v); Chapter 4, page 136
///
/// Default values for a and b were proposed by (LeCun, 1993),
/// so that phi(1) = 1 and phi(-1) = -1, and the slope at the origin is 1.1424;
/// Chapter 4, page 145.
class TanhActivation : public ActivationFunction {
private:
    float a;
    float b;

public:
    TanhActivation(float a = 1.7159, float b = 0.6667) : a(a), b(b) {}

    virtual float operator()(float v) const { return a * tanh(b * v); }

    virtual float derivative(float v) const {
        float y = tanh(b * v);
        return a * b * (1.0 - y * y);
    }

    virtual void print(std::ostream &out) const { out << "tanh"; }
};


class PiecewiseLinearActivation : public ActivationFunction {
private:
    float negativeSlope;
    float positiveSlope;
    std::string name;

public:
    PiecewiseLinearActivation(float negativeSlope = 0.0,
                              float positiveSlope = 1.0,
                              std::string name = "ReLU")
        : negativeSlope(negativeSlope), positiveSlope(positiveSlope), name(name) {}

    virtual float operator()(float v) const {
        if (v >= 0) {
            return positiveSlope * v;
        } else {
            return negativeSlope * v;
        }
    }

    virtual float derivative(float v) const {
        if (v >= 0) {
            return positiveSlope;
        } else {
            return negativeSlope;
        }
    }

    virtual void print(std::ostream &out) const { out << name; }
};


const TanhActivation defaultTanh(1.0, 1.0);
const TanhActivation scaledTanh; //< tanh with LeCun parameters
const SignumActivation defaultSignum;
const PiecewiseLinearActivation ReLU;
const PiecewiseLinearActivation leakyReLU(0.01f, 1.0f, "leakyReLU");
const PiecewiseLinearActivation linearActivation(1.0f, 1.0f, "");


/**
 * Rosenblatt's Perceptron
 * -----------------------
 **/

/// A stand-alone neuron with adjustable synaptic weights and bias.
class LinearPerceptron : public ANeuron {
private:
    Weights weights;
    const ActivationFunction &activationFunction;

public:
    LinearPerceptron(int n, const ActivationFunction &af = linearActivation)
        : weights(n + 1), activationFunction(af) {}

    virtual float inducedLocalField(const Input &x) {
        assert(x.size() + 1 == weights.size());
        float bias = weights[0];
        auto begin_weights = std::next(std::begin(weights));
        return std::inner_product(begin_weights, std::end(weights), std::begin(x), bias);
    }

    virtual float output(const Input &x) {
        return activationFunction(inducedLocalField(x));
    }

    virtual Weights getWeights() const {
        return weights;
    }

    virtual Weights adjustWeights(const Weights &weightCorrections) {
        assert(weights.size() == weightCorrections.size());
        weights += weightCorrections;
        return weights;
    }
};


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
void trainConverge_addSample(ANeuron &p, Input input, float output, float eta) {
    float y = sign(p.output(input));
    float xfactor = eta * (output - y);
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
                   int epochs, float eta) {
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
void trainBatch_addBatch(ANeuron &p, LabeledDataset batch, float eta) {
    for (auto sample : batch) {
        float desired = sample.label[0]; // desired output
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

void trainBatch(ANeuron &p, const LabeledDataset &trainSet, int epochs, float eta) {
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
 * Multilayer Perceptrons
 * ----------------------
 **/

/// An artificial neuron with back-propagation capability.
class BidirectionalNeuron : public ANeuron {
private:
    int nInputs;
    Weights weights; // weights[0] is bias
    const ActivationFunction &activationFunction;

    // remember the latest internal parameters to use them
    // again in the back-propagation step
    float lastInducedLocalField;  // v_j = \sum w_i y_i
    float lastActivationValue;    // y_j = \phi (v_j)
    float lastActivationGradient; // dy_j = \phi^\prime (v_j)
    float lastLocalGradient;      // delta_j = \phi^\prime(v_j) e_j

public:
    BidirectionalNeuron(int n, const ActivationFunction &af = scaledTanh)
        : nInputs(n), weights(n + 1), activationFunction(af) {}

    void init(std::unique_ptr<RNG> &rng, WeightsInitializer init_fn) {
        init_fn(rng, weights);
    }

    virtual float inducedLocalField(const Input &x) {
        auto bias = weights[0];
        auto begin_weights = std::next(std::begin(weights));
        return std::inner_product(begin_weights, std::end(weights), std::begin(x), bias);
    }

    virtual float output(const Input &x) {
        float v = inducedLocalField(x);
        lastInducedLocalField = v;
        lastActivationValue = activationFunction(v);
        lastActivationGradient = activationFunction.derivative(v);
        return lastActivationValue;
    }

    virtual Weights getWeights() const {
        return weights;
    }

    virtual Weights adjustWeights(const Weights &weightCorrections) {
        assert(weights.size() == weightCorrections.size());
        weights += weightCorrections;
        return weights;
    }

    struct BackOutput {
        float localGradient;
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
    // Return a vector of weight corrections (not multiplied by the learning
    // rate yet) and the local gradient value.
    //
    // The method should be called _after_ `forwardPass`
    BackOutput backwardPass(const Input &inputs, float errorSignal) {
        float localGradient = lastActivationGradient * errorSignal;
        Weights delta_W(localGradient, weights.size());
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

public:
    FullyConnectedLayer(unsigned int nInputs = 0, unsigned int nOutputs = 0,
                     const ActivationFunction &af = scaledTanh)
        : nInputs(nInputs), nNeurons(nOutputs),
          neurons(nOutputs, BidirectionalNeuron(nInputs, af)) {}

    void init(std::unique_ptr<RNG> &rng, WeightsInitializer init_fn) {
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].init(rng, init_fn);
        }
    }

    void adjustWeights(std::vector<Weights> weightDeltas, float learningRate) {
        assert(nNeurons == weightDeltas.size());
        for (size_t i = 0; i < neurons.size(); ++i) {
            neurons[i].adjustWeights(weightDeltas[i] * learningRate);
        }
    }

    // Pages 132-133.
    // Return a vector of outputs.
    Output forwardPass(const Input &inputs) {
        Output outputs(nNeurons);
        return forwardPass(inputs, outputs);
    }

    Output &forwardPass(const Input &inputs, Output &outputs) {
        for (auto i = 0u; i < nNeurons; ++i) {
            outputs[i] = neurons[i].output(inputs);
        }
        return outputs;
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
    BackOutput backwardPass(const Input &inputs, const Output &errorSignals) {
        assert(errorSignals.size() == neurons.size());
        std::vector<Weights> weightDeltas(0);
        Weights propagatedErrorSignals(0.0, nInputs);
        for (auto k = 0u; k < nNeurons; ++k) {
            auto error_k = errorSignals[k];
            auto r = neurons[k].backwardPass(inputs, error_k);
            Weights delta_Wk = r.weightCorrections;
            float delta_k = r.localGradient;
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
                         const ActivationFunction &af = scaledTanh)
        : layers(0), layersInputs(0) {
        assert(shape.size() > 0);
        auto pIn = shape.begin();
        auto pOut = std::next(pIn);
        for (; pOut != shape.end(); ++pIn, ++pOut) {
            auto inSize = *pIn;
            auto outSize = *pOut;
            FullyConnectedLayer layer(inSize, outSize, af);
            layers.push_back(layer);
            // the corresponding input buffer
            Input input(0.0, inSize);
            layersInputs.push_back(input);
        }
        // output of the last layer
        auto outSize = *pIn;
        Input output(0.0, outSize);
        layersInputs.push_back(output);
    }

    void init(std::unique_ptr<RNG> &rng,
              WeightsInitializer init_fn = normalXavier) {
        for (auto i = 0u; i < layers.size(); ++i) {
            layers[i].init(rng, init_fn);
        }
    }

    Output &forwardPass(const Input &inputs) {
        assert(layers.size() > 0);
        layersInputs[0] = inputs;
        for (auto i = 0u; i < layers.size(); ++i) {
            Input &in = layersInputs[i];
            Input &out = layersInputs[i+1];
            layers[i].forwardPass(in, out);
        }
        return layersInputs[layers.size()];
    }

    // TODO: unify backwardPass intefrace of the neuron, layer and network
    FullyConnectedLayer::BackOutput
    backwardPass(const Output &errorSignals, float learningRate) {
        Output err(errorSignals);
        FullyConnectedLayer::BackOutput r;

        for (int i = layers.size()-1; i >= 0; --i) {
            auto layerIn = layersInputs[i];
            r = layers[i].backwardPass(layerIn, err);
            layers[i].adjustWeights(r.weightCorrections, learningRate);
            err = r.propagatedErrorSignals;
        }
        return r;
    }

    friend std::ostream &operator<<(std::ostream &out, const MultilayerPerceptron &net);
};


/**
 * Loss Functions
 * --------------
 *
 **/

using LossFunction = std::function<float(const Output &, const Output &)>;

/** Euclidean loss.
 *
 * $$ E_2(\mathbf{a}, \mathbf{b}) = \sqrt{ \sum_i (a_i - b_i)^2 } $$
 *
 * It may be used for regressing real-valued labels.
 * https://en.wikipedia.org/wiki/Convolutional_neural_network#Loss_layer */
float L2_loss(const Output &actualOutput, const Output &expectedOutput) {
    float loss2 = std::inner_product(
            std::begin(actualOutput), std::end(actualOutput),
            std::begin(expectedOutput),
            0.0,
            [](float s_i, float s_inext) { return s_i + s_inext; },
            [](float a_i, float b_i) { return (a_i - b_i)*(a_i - b_i); });
    return sqrt(loss2);
}

float totalLoss(LossFunction loss,
                MultilayerPerceptron &net,
                const LabeledDataset& testSet) {
    float totalLoss = 0.0;
    for (auto sample : testSet) {
        auto out = net.forwardPass(sample.data);
        totalLoss += loss(out, sample.label);
    }
    return totalLoss;
}

#endif /* NOTCH_H */
