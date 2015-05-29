#ifndef NOTCH_H
#define NOTCH_H

// TODO: doxygen-compatible comments
// TODO: optional OpenMP implementation
// TODO: benchmarks
// TODO: tests

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

#include <algorithm>  // generate, transform
#include <array>      // array
#include <cmath>      // sqrt, exp
#include <functional> // ref, function<>
#include <initializer_list>
#include <iomanip>    // setw, setprecision
#include <iterator>   // begin, end
#include <memory>     // unique_ptr, make_unique
#include <numeric>    // inner_product
#include <ostream>    // ostream
#include <random>
#include <string>
#include <typeinfo>   // typeid
#include <type_traits> // enable_if, is_pointer
#include <valarray>
#include <vector>

#ifdef NOTCH_USE_CBLAS
#include <cblas.h>
#endif

/**
 * Library Framework
 * =================
 **/

/**
 * Random number generation
 * ------------------------
 */

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
using Array = std::valarray<float>;
using Input = Array;
using Output = Array;

// TODO: NDArray, mostly compatible with Array; like struct { shape; data; };

/** Unlabeled dataset are collections or `Input`s or `Output`s. */
using Dataset = std::vector<Array>;


/** A common interface to preprocess data.
 *  Transformations are supposed to be adaptive (`fit`) and invertable. */
class ADatasetTransformer {
public:
    virtual ADatasetTransformer& fit(const Dataset &) = 0;
    virtual Dataset transform(const Dataset &) = 0;
    virtual Array transform(const Array &) = 0;
    virtual Dataset inverse_transform(const Dataset &) = 0;
    virtual Array inverse_transform(const Array &) = 0;
};


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
    Dataset inputs;
    Dataset outputs;

public:

    /// An iterator type to process all labeled data samples.
    class DatasetIterator : public std::iterator<std::input_iterator_tag, LabeledData> {
    private:
        using ArrayVecIter = Dataset::const_iterator;
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

    const Dataset &getData() { return inputs; }

    const Dataset &getLabels() { return outputs; }

    /// Preprocess `Input` data
    void transform(ADatasetTransformer &t) {
        inputs = t.transform(inputs);
        inputDimension = inputs.size();
    }

    /// Preprocess `Output` labels
    void transformLabels(ADatasetTransformer &t) {
        outputs = t.transform(outputs);
        outputDimension = outputs.size();
    }

    /// Randomly shuffle `LabeledData`.
    void shuffle(std::unique_ptr<RNG> &rng) {
        // Modern version of Fischer-Yates shuffle.
        // The same random permutation should be applied to both
        // inputs and outputs, so std::shuffle is not applicable.
        size_t n = inputs.size();
        for (size_t i = n - 1; i >= 1; --i) {
            std::uniform_int_distribution<> ud(0, i);
            size_t j = ud(*rng);
            if (i != j) {
                std::swap(inputs[i], inputs[j]);
                std::swap(outputs[i], outputs[j]);
            }
        }
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

// TODO: maybe rename to WeightsInit
using WeightInit =
    std::function<void(std::unique_ptr<RNG> &, Weights &, int, int)>;

/** One-sided Xavier initialization.
 *
 * Pick weights from a zero-centered _normal_ distrubition with variance
 * $$\sigma^2 = 1/n_{in}$$, where $n_{in}$ is the number of inputs.
 *
 * See
 *
 *  - NNLM3, Chapter 4, page 149;
 *  - http://andyljones.tumblr.com/post/110998971763/ **/
void normalXavier(std::unique_ptr<RNG> &rng, Weights &weights, int n_in, int) {
    float sigma = n_in > 0 ? sqrt(1.0 / n_in) : 1.0;
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
 * See
 *
 *  - NNLM3, Chapter 4, page 149;
 *  - http://andyljones.tumblr.com/post/110998971763/ **/
void uniformXavier(std::unique_ptr<RNG> &rng, Weights &weights, int n_in, int) {
    float sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
    float a = sigma * sqrt(3.0);
    std::uniform_real_distribution<float> nd(-a, a);
    std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
                float w = nd(*rng.get());
                return w;
             });
}

/**
 * Activation Functions
 * --------------------
 **/

 /** Activation functions are applied to neuron's output to introduce
 * non-linearity and map output to specific range. Backpropagation algorithm
 * requires differentiable activation functions. */
class ActivationFunction {
public:
    virtual float operator()(float v) const = 0;
    virtual float derivative(float v) const = 0;
    virtual void print(std::ostream &out) const = 0;
};


/** Logistic function maps output to the range (0,1).
 *
 * It is defined as
 *
 * $$\phi(v) = 1/(1 + \exp(- s v)),$$
 *
 * where $s$ is its derivative at zero.
 * See  NNLM3, Chapter 4, page 135. */
class LogisticActivation : public ActivationFunction {
private:
    float slope = 1.0;

public:
    LogisticActivation(float slope) : slope(slope) {};

    virtual float operator()(float v) const {
        return 1.0 / (1.0 + exp(-slope * v));
    }

    virtual float derivative(float v) const {
        float y = (*this)(v);
        return slope * y * (1 - y);
    }

    virtual void print(std::ostream &out) const {
        out << "logistic";
        if (slope != 1.0) {
            out << "(" << slope << ")";
        }
    }
};


/** Hyperbolic tangent activation function.
 *
 * It is defined as $\phi(v) = a  \tanh(b v)$, NNLM3, Chapter 4, page 136.
 * It maps output to the range (-a, a), or (-1, 1) by default.
 *
 * Yann LeCun proposed parameters $a = 1.7159$ and $b = 2/3$,
 * so that $\phi(1) = 1$ and $\phi(-1) = -1$, and the slope at the origin is
 * close to one (1.1424). scaledTanh uses this parameters.
 *
 * References:
 *
 * - NNLM3, Chapter 4, page 145;
 * - Y. LeCun (1989) Generalization and Network Design Strategies. page 7;
 * - Y. LeCun et al. (2012) Efficient BackProp
 */
class TanhActivation : public ActivationFunction {
private:
    float a;
    float b;
    std::string name;

public:
    TanhActivation(float a, float b, std::string name = "tanh")
        : a(a), b(b), name(name) {}

    virtual float operator()(float v) const { return a * tanh(b * v); }

    virtual float derivative(float v) const {
        float y = tanh(b * v);
        return a * b * (1.0 - y * y);
    }

    virtual void print(std::ostream &out) const { out << name; }
};


/** Piecewise linear activation function allows to implement
 * rectified linear units (ReLU) and leaky rectified linear units (leakyReLU).
 *
 * Rectifiers are often used in deep neural networks because they don't
 * suffer from diminishing gradients problem and allow for sparse activation
 * (neurons with negative induced local field remain dormant). */
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


/// Default hyperbolic tangent activation.
const TanhActivation defaultTanh(1.0, 1.0, "tanh");
/// Hyperbolic tangent activation with LeCun parameters.
const TanhActivation scaledTanh(1.7159, 0.6667, "scaledTanh");
/// Logistic function activation (output values between 0 and 1).
const LogisticActivation logisticActivation(1.0f);
/// Rectified Linear Unit: $\phi(v) = \max(0, v)$.
const PiecewiseLinearActivation ReLU(0.0f, 1.0f, "ReLU");
/// Rectified Linear Unit with small non-zero gradient in inactive zone:
/// $\phi(v) = v \text{if} v > 0 \text{or} 0.01 v \text{otherwise}$.
const PiecewiseLinearActivation leakyReLU(0.01f, 1.0f, "leakyReLU");
/// Linear activation.
const PiecewiseLinearActivation linearActivation(1.0f, 1.0f, "linear");


/**
 * Multilayer Perceptrons
 * ----------------------
 **/

// TODO: FixedRate $\eta = \mathrm{const}$
// TODO: AdaptiveRate $\eta ~ 1/\sqrt{n_{in}}$ (NNLM3, page 150; (LeCun, 1993))
// TODO: update with momentum (ASGD)

/** Values calculated in the backpropagation step.
 *
 * propagatedErrors $e_i$ is blame assigned to each of the inputs $i$;
 *
 * weightSensitivy $\partial E/\partial w{ji}$ defines direction of the weight
 * corrections in the delta rule; biasSensitivity $\partial E/\partial b{j}$
 * does the same for bias corrections.
 *
 * The actual delta rule is define
 */
struct BackpropResult {
    BackpropResult(size_t nInputs, size_t nOutputs)
        : propagatedErrors(0.0, nInputs),
          weightSensitivity(0.0, nInputs*nOutputs),
          biasSensitivity(0.0, nOutputs) {}
    Array propagatedErrors; //< $e_i$
    Array weightSensitivity; //< $\partial E/\partial w_{ji}$
    Array biasSensitivity; //< $\partial E/\partial b{j}$
};

/** A base class for the rule to correct weights and bias given sensitivity factors.
 *
 * Concrete implementations may overwrite weightSensitivity and biasSensitivity
 * parameters. Output is written to weighs and bias parameters. */
class ALearningPolicy {
public:
    virtual void correctWeights(Array& weightSensitivy, Array &weights) = 0;
    virtual void correctBias(Array& biasSensitivity, Array &bias) = 0;
    virtual void resize(size_t nInputs, size_t nOutputs) = 0;
    virtual std::unique_ptr<ALearningPolicy> clone() const = 0;
};

/** A classic delta rule.
 *
 * $w_{ji} (n) = w_{ji} (n-1) + \Delta w_{ji}$, where
 * $\Delta w_{ji} = - \eta \partial E / \partial w_{ji}$.
 *
 * Reference: NNLM3, Chapter 4, page 131; (LeCun, 2012), page 12. */
class FixedRate : public ALearningPolicy {
private:
    float learningRate;
public:
    FixedRate(float learningRate = 0.01) : learningRate(learningRate) {}
    virtual void correctWeights(Array& weightSensitivity, Array &weights) {
        weights -= (learningRate * weightSensitivity);
    }
    virtual void correctBias(Array& biasSensitivity, Array &bias) {
        bias -= (learningRate * biasSensitivity);
    }
    virtual void resize(size_t, size_t) {}
    virtual std::unique_ptr<ALearningPolicy> clone() const {
        auto c = std::unique_ptr<ALearningPolicy>(new FixedRate(learningRate));
        return c;
    }
};

/** Generalized delta rule for learning with _momentum_ term.
 *
 * $$\Delta w_{ji} (n) = \alpha \Delta w_{ji} (n-1)
 *                     - \eta \partial E / \partial w_{ji},$$
 *
 * where $\eta$ is a learning rate, and $\alpha$ is a momentum constant.
 *
 * The momentum term tends to accelerate descent in steady downhill dirctions
 * when the partial derivative $\partial E/\partial w_{ji}$ has the same sign
 * on consecutive iterations. It has stabilizing effect otherwise.
 *
 * References:
 *
 *  - NNLM3, Eq (4.41), page 137
 *  - LeCun (2012) Efficient Backprop, page 21
 */
class FixedRateWithMomentum : public ALearningPolicy {
private:
    float learningRate;
    float momentum;
    Array lastDeltaW;
    Array lastDeltaB;
public:
    FixedRateWithMomentum(float learningRate = 0.01, float momentum = 0.9)
        : learningRate(learningRate), momentum(momentum) {}
    virtual void correctWeights(Array& weightSensitivity, Array &weights) {
        if (lastDeltaW.size() != weights.size()) {
            lastDeltaW.resize(weights.size(), 0.0);
        }
        lastDeltaW = (momentum * lastDeltaW - learningRate * weightSensitivity);
        weights += lastDeltaW;
    }
    virtual void correctBias(Array& biasSensitivity, Array &bias) {
        if (lastDeltaB.size() != bias.size()) {
            lastDeltaB.resize(bias.size(), 0.0);
        }
        lastDeltaB = (momentum * lastDeltaB - learningRate * biasSensitivity);
        bias += lastDeltaB;
    }
    virtual void resize(size_t nInputs, size_t nOutputs) {
        lastDeltaW.resize(nInputs * nOutputs, 0.0);
        lastDeltaB.resize(nOutputs, 0.0);
    }
    virtual std::unique_ptr<ALearningPolicy> clone() const {
        auto c = std::unique_ptr<ALearningPolicy>(
                  new FixedRateWithMomentum(learningRate, momentum));
        return c;
    }
};

class ABackpropLayer;

// TODO: maybe rename inputDim, outputDim to inputSize(), outputSize() (?)
/** Get and set layer's parameters. */
class ALayer {
public:
    /// A name to identify layer type.
    virtual std::string tag() const = 0;

    /// Randomly initialize synaptic weights.
    virtual void init(std::unique_ptr<RNG> &rng, WeightInit init) = 0;
    /// Initialize synaptic weights using a weights matrix.
    virtual void init(Array &&weights, Array &&bias) = 0;
    /// Initialize synaptic weights using a copy of a weights matrix.
    virtual void init(const Array &weights, const Array &bias) = 0;

    /// Share output buffers with the nextLayer.
    virtual void connectTo(ALayer& nextLayer) = 0;
    virtual std::shared_ptr<Array> &getInputBuffer() = 0;
    virtual std::shared_ptr<Array> &getOutputBuffer() = 0;
    virtual std::shared_ptr<ABackpropLayer> clone() = 0;

    /// Get the number of input variables.
    virtual size_t inputDim() const = 0;
    /// Get the number of output variables.
    virtual size_t outputDim() const = 0;
    /// Get the synaptic weights matrix.
    virtual const Array &getWeights() const = 0;
    /// Get the output bias matrix.
    virtual const Array &getBias() const = 0;

    /// Set layer's activation function.
    virtual void setActivationFunction(const ActivationFunction &) = 0;
    /// Get layer's activation function.
    virtual const ActivationFunction &getActivationFunction() const = 0;
};

/** A common interface of all layers or groups of layers (networks) capable of
 * both forward and backpropagation.
 *
 * Output results (Array) can be shared between layers, because
 * the output of the layer $j$ is the input of the layer $j+1$.
 * Backpropagation results (BackpropResult) can be shared between
 * layers too.*/
class ABackpropNet {
public:
    /// a forward propagaiton pass
    virtual std::shared_ptr<Array> output(const Array &inputs) = 0;
    /// a backpropagation pass
    virtual std::shared_ptr<BackpropResult> backprop(const Array &errors) = 0;
    /// specify how the weights have to be adjusted
    virtual void setLearningPolicy(const ALearningPolicy &lp) = 0;
    /// adjust layer weights after the backpropagation pass
    virtual void update() = 0;
};

class ABackpropLayer : public ALayer, public ABackpropNet {};

#ifdef NOTCH_USE_CBLAS

/** Matrix-vector product using CBLAS.
 * Calculate $M*x + b$ and save result to $b$.
 *
 * Note: CBLAS requires pointers to data.
 * The type of std::begin(std::valarray&) is not specified by the standard,
 * but GNU and Clang implementations return a plain pointer.
 * We can enable interoperation with CBLAS only for compilers like GNU. */
template <class Matrix_Iter, class VectorX_Iter, class VectorB_Iter>
typename std::enable_if<std::is_pointer<Matrix_Iter>::value &&
                        std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorB_Iter>::value, void>::type
gemv(Matrix_Iter m_begin, Matrix_Iter m_end,
     VectorX_Iter x_begin, VectorX_Iter x_end,
     VectorB_Iter b_begin, VectorB_Iter b_end) {
    size_t cols = std::distance(x_begin, x_end);
    size_t rows = std::distance(b_begin, b_end);
    size_t n = std::distance(m_begin, m_end);
    if (n != rows * cols) {
        std::string what = "blas_gemv: incompatible matrix and vector shapes";
        throw std::invalid_argument(what);
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                rows, cols,
                1.0 /* M multiplier */,
                m_begin, cols /* leading dimension of M */,
                x_begin, 1,
                1.0 /* b multiplier */,
                b_begin, 1);
}

#else /* NOTCH_USE_CBLAS is not defined */

/** Matrix-vector product on STL iterators, similar to BLAS _gemv function.
 * Calculate $M x + b$ and saves result in $b$. */
template <class Matrix_Iter, class VectorX_Iter, class VectorB_Iter>
void
gemv(Matrix_Iter m_begin, Matrix_Iter m_end,
     VectorX_Iter x_begin, VectorX_Iter x_end,
     VectorB_Iter b_begin, VectorB_Iter b_end) {
    size_t cols = std::distance(x_begin, x_end);
    size_t rows = std::distance(b_begin, b_end);
    size_t n = std::distance(m_begin, m_end);
    if (n != rows * cols) {
        std::string what = "stl_gemv: incompatible matrix and vector shapes";
        throw std::invalid_argument(what);
    }
    size_t r = 0; // current row number
    for (auto b = b_begin; b != b_end; ++b, ++r) {
        auto row_begin = m_begin + r * cols;
        auto row_end = row_begin + cols;
        *b = std::inner_product(row_begin, row_end, x_begin, *b);
    }
}

#endif /* ifdef NOTCH_USE_CBLAS */


/** A fully connected layer of neurons with backpropagation. */
class FullyConnectedLayer : public ABackpropLayer {
protected:
    size_t nInputs;
    size_t nOutputs; //< the number of neurons in the layer
    Array weights; //< weights matrix $w_ji$ for the entire layer, row-major order
    Array bias;    //< bias values $b_j$ for the entire layer, one per neuron
    const ActivationFunction *activationFunction;

    Array inducedLocalField;            //< $v_j = \sum_i w_ji x_i + b_j$
    Array activationGrad;               //< $dy/dv_j = \phi^\prime (v_j)$
    Array localGrad;                    //< $\delta_j = dE/dv_j = \phi^\prime(v_j) e_j$

    // Forward and backpropagation results can be shared between layers.
    // The buffers are allocated once either in init() or at the begining of the
    // forward or backprop pass respectively (see output() and backprop()).
    // Actual allocation is implemented in allocateInOutBuffers() method,
    // connectTo() method implements sharing.
    std::shared_ptr<Array> lastInputs;  //< $x_i$
    std::shared_ptr<Array> lastOutputs; //< $y_j = \phi(v_j)$
    std::shared_ptr<BackpropResult> thisBPR; //< backpropagation results of this layer
    bool buffersAreReady = false; //< true if in/out and backprop buffers are allocated

    std::shared_ptr<ALearningPolicy> policy;

    void rememberInputs(const Array& inputs) {
        *lastInputs = inputs;  // remember for calcSensitivityFactors()
    }

    /** Linear response of the layer $v_j = w_{ji} x_i$. */
    void calcInducedLocalField(const Array &inputs) {
        inducedLocalField = bias; // will be added and overwritten
        gemv(std::begin(weights), std::end(weights), std::begin(inputs),
             std::end(inputs), std::begin(inducedLocalField),
             std::end(inducedLocalField));
    }

    /** Derivatives of the activation functions. */
    void calcActivationGrad() {
        std::transform(
            std::begin(inducedLocalField), std::end(inducedLocalField),
            std::begin(activationGrad),
            [&](float y) { return activationFunction->derivative(y); });
    }

    /** Non-linear response of the layer $y_j = \phi(v_j)$. */
    void calcOutput(Array &outputs) {
        std::transform(std::begin(inducedLocalField),
                       std::end(inducedLocalField),
                       std::begin(outputs),
                       [&](float y) { return (*activationFunction)(y); });
    }

    void outputInplace(const Array &inputs, Array &outputs) {
        if (outputs.size() != nOutputs) {
            outputs.resize(nOutputs);
        }
        rememberInputs(inputs);
        calcInducedLocalField(inputs);
        calcActivationGrad();
        calcOutput(outputs);
    }

    /** The local gradient $\delta_j = \partial E/\partial v_j$ is the product
     * of the activation function derivative and the error signal. */
    void calcLocalGrad(const Array &errors) {
        assert(activationGrad.size() == errors.size());
        assert(localGrad.size() == errors.size());
        localGrad = activationGrad * errors;
    }

    /** Calculate sensitivity factors $\partial E/\partial w_{ji}$ and
     * $\partial E/\partial b_j$ for weights and bias respectively.
     *
     * The derivative are then used to corrections to the matrix of the
     * synaptic weights and biases.
     *
     * NNLM3, Page 134, Eq. (4.27) defines weight correction as
     *
     *  $$ \Delta w_{ji} (n) = \eta \times \delta_j (n) \times y_{i} (n) $$
     *
     * where $w_{ji}$ is the synaptic weight connecting neuron $i$ to neuron $j$,
     * $\eta$ is learning rate, $delta_j (n)$ is the local [error] gradient
     * $\partial E (n)/\partial v_j (n)$, $y_{i}$ is the input signal of the
     * neuron $i$, $n$ is the epoch number We discard $\eta$ at the moment
     * (because it is part of ALearningPolicy).
     *
     * Or in terms of the sensitivity factor, Eq. (4.14) idem:
     *
     * $$ \Delta w_{ji} (n) = - \eta \partial E(n) / \partial w_{ji}(n)$$
     *
     * Hence, at this point we may calculate the sensitivity factor
     *
     * $$dE/dw_ji = - \delta_j y_i$$ */
    void calcSensitivityFactors(Array &weightSensitivity, Array &biasSensitivity) {
        assert(weightSensitivity.size() == weights.size());
        assert(biasSensitivity.size() == bias.size());
        assert(localGrad.size() == nOutputs);
        assert(lastInputs->size() == nInputs);
        for (size_t j = 0; j < nOutputs; ++j) { // for all neurons (rows)
            for (size_t i = 0; i < nInputs; ++i) { // for all inputs (columns)
                float y_i = (*lastInputs)[i];
                weightSensitivity[j*nInputs + i] = (-1.0 * localGrad[j] * y_i);
            }
            biasSensitivity[j] = (-1.0 * localGrad[j]);
        }
    }

    /** Calculate back-propagated error signal and corrections to synaptic weights.
    * NNLM3r, Page 134.
    *
    * $$ e_j = \sum_k \delta_k w_{kj} $$
    *
    * where $e_j$ is an error propagated from all downstream neurons to the
    * neuron $j$, $\delta_k$ is the local gradient of the downstream neurons
    * $k$, $w_{kj}$ is the synaptic weight of the $j$-th input of the
    * downstream neuron $k$. */
    void calcPropagatedErrors(Array &propagatedErrors) {
        if (propagatedErrors.size() != nInputs) {
            propagatedErrors.resize(nInputs);
        }
        for (size_t j = 0; j < nInputs; ++j) { // for all inputs
            float e_j = 0.0;
            for (size_t k = 0; k < nOutputs; ++k) { // for all neurons
                e_j += localGrad[k] * weights[k*nInputs + j];
            }
            propagatedErrors[j] = e_j;
        }
    }

    /// Backpropagation algorithm
    void
    backpropInplace(const Array &errors, BackpropResult &bp) {
        calcLocalGrad(errors);
        calcSensitivityFactors(bp.weightSensitivity, bp.biasSensitivity);
        calcPropagatedErrors(bp.propagatedErrors);
    }

    /// Allocates lastInputs and lastOutputs buffers if they're not allocated yet.
    void allocateInOutBuffers() {
        if (buffersAreReady) {
            return;
        }
        if (!lastInputs) {
            lastInputs = std::make_shared<Array>(0.0, nInputs);
        }
        if (!lastOutputs) {
            lastOutputs = std::make_shared<Array>(0.0, nOutputs);
        }
        if (!thisBPR) {
            thisBPR = std::make_shared<BackpropResult>(nInputs, nOutputs);
        }
        buffersAreReady = true;
    }

    /// @return true if input-output buffers are allocated _and_ shared
    bool isConnected() const {
        return (buffersAreReady &&
                !(lastInputs.unique() &&
                  lastOutputs.unique() &&
                  thisBPR.unique()));
    }

    /// Resize all layer buffers if it is initialized with a weight matrix
    /// of different shape. We can do it only if the layer is not connected
    /// to other layer (is not sharing buffers), i.e. before connectTo().
    void initResize(const Weights& weights, const Weights &bias) {
        bool needResize = this->weights.size() != weights.size() ||
                          this->bias.size() != bias.size();
        if (needResize) {
            if (isConnected()) {
                throw std::invalid_argument("cannot reshape a connected layer");
            } else {
                size_t n_in = weights.size() / bias.size();
                size_t n_out = bias.size();
                if (n_in * n_out != weights.size()) {
                    throw std::invalid_argument("incompatible weights/bias shapes");
                }
                // resize everything
                nInputs = n_in;
                nOutputs = n_out;
                this->weights.resize(n_in * n_out, 0.0);
                this->bias.resize(n_out);
                inducedLocalField.resize(n_out);
                activationGrad.resize(n_out);
                localGrad.resize(n_out);
                // resize shared buffers which may happen to be allocated
                // in the stand-alone (disconnected) layer
                if (lastInputs) {
                    lastInputs->resize(n_in);
                }
                if (lastOutputs) {
                    lastOutputs->resize(n_out);
                }
                if (thisBPR) {
                    thisBPR = std::make_shared<BackpropResult>(n_in, n_out);
                }
                // resize buffers for historical values
                if (policy) {
                    policy->resize(n_in, n_out);
                }
           }
        }
    }

public:
    /// Create a layer with zero weights.
    // TODO: initialize policy
    FullyConnectedLayer(size_t nInputs = 0, size_t nOutputs = 0,
                        const ActivationFunction &af = scaledTanh)
        : nInputs(nInputs), nOutputs(nOutputs),
          weights(nInputs * nOutputs), bias(nOutputs), activationFunction(&af),
          inducedLocalField(nOutputs), activationGrad(nOutputs), localGrad(nOutputs),
          // shared buffers are allocated dynamically
          lastInputs(nullptr), lastOutputs(nullptr),
          thisBPR(nullptr) {}

    /// Create a layer from a weights matrix.
    FullyConnectedLayer(Weights &&weights, Weights &&bias,
                        const ActivationFunction &af = scaledTanh)
        : nInputs(weights.size()/bias.size()),
          nOutputs(bias.size()),
          weights(weights), bias(bias), activationFunction(&af),
          inducedLocalField(nOutputs), activationGrad(nOutputs), localGrad(nOutputs),
          // shared buffers are allocated dynamically
          lastInputs(nullptr), lastOutputs(nullptr),
          thisBPR(nullptr) {}

    /// Create a layer from a copy of a weights matrix.
    FullyConnectedLayer(const Weights &weights, const Weights &bias,
                        const ActivationFunction &af = scaledTanh)
        : nInputs(weights.size()/bias.size()),
          nOutputs(bias.size()),
          weights(weights), bias(bias), activationFunction(&af),
          inducedLocalField(nOutputs), activationGrad(nOutputs), localGrad(nOutputs),
          // shared buffers are allocated dynamically
          lastInputs(nullptr), lastOutputs(nullptr),
          thisBPR(nullptr) {}

    /* begin ALayer interface */
    virtual std::string tag() const { return "FullyConnectedLayer"; }

    /// Initialize synaptic weights.
    virtual void init(std::unique_ptr<RNG> &rng, WeightInit init) {
        init(rng, weights, nInputs, nOutputs);
        init(rng, bias, nInputs, nOutputs);
    }

    /// Initialize synaptic weights.
    virtual void init(Weights &&weights, Weights &&bias) {
        // allow changing size of not-yet-connected layers
        initResize(weights, bias);
        this->weights = weights;
        this->bias = bias;
    }

    /// Initialize synaptic weights.
    virtual void init(const Weights &weights, const Weights &bias) {
        // allow changing size of not-yet-connected layers
        initResize(weights, bias);
        this->weights = weights;
        this->bias = bias;
    }

    /// Interlayer connections allow to share input-output buffers between two layers.
    virtual void connectTo(ALayer& nextLayer) {
        if (nextLayer.inputDim() != this->outputDim()) {
            throw std::invalid_argument("incompatible shape of the nextLayer");
        }
        allocateInOutBuffers();
        nextLayer.getInputBuffer() = this->lastOutputs;
    }

    virtual std::shared_ptr<Array> &getInputBuffer() {
        return lastInputs;
    }

    virtual std::shared_ptr<Array> &getOutputBuffer() {
        return lastOutputs;
    }

    virtual std::shared_ptr<ABackpropLayer> clone() {
        auto p = std::make_shared<FullyConnectedLayer>(
                nInputs, nOutputs, *activationFunction);
        if (!p) {
            return p;
        }
        p->weights = weights;
        p->bias = bias;
        p->inducedLocalField = inducedLocalField;
        p->activationGrad = activationGrad;
        p->localGrad = localGrad;
        if (buffersAreReady) { // clone shared buffers too
            p->allocateInOutBuffers();
            if (p->lastInputs && lastInputs) {
                *(p->lastInputs) = *(lastInputs);
            }
            if (p->lastOutputs && lastOutputs) {
                *(p->lastOutputs) = *(lastOutputs);
            }
            if (p->thisBPR && thisBPR) {
                *(p->thisBPR) = *(thisBPR);
            }
        }
        if (policy) {
            p->policy = policy->clone();
        }
        return p;
    }

    virtual size_t inputDim() const {
        return nInputs;
    }
    virtual size_t outputDim() const {
        return nOutputs;
    }
    virtual const Weights &getWeights() const {
        return weights;
    }
    virtual const Weights &getBias() const {
        return bias;
    }
    virtual void setActivationFunction(const ActivationFunction &af) {
        activationFunction = &af;
    }
    virtual const ActivationFunction &getActivationFunction() const {
        return *activationFunction;
    }
    /* end ALayer interface */

    /** begin ABackpropNet interface */
    virtual std::shared_ptr<Array> output(const Array &inputs) {
        allocateInOutBuffers(); // just in case the user didn't init()
        outputInplace(inputs, *lastOutputs);
        return lastOutputs;
    }

    virtual std::shared_ptr<BackpropResult> backprop(const Array &errors) {
        allocateInOutBuffers(); // just in case the user didn't init()
        backpropInplace(errors, *thisBPR);
        return thisBPR;
    }

    virtual void setLearningPolicy(const ALearningPolicy &lp) {
        policy = lp.clone();
    }

    virtual void update() {
        if (!policy) {
            throw std::logic_error("learning policy is not defined");
        }
        policy->correctWeights(thisBPR->weightSensitivity, weights);
        policy->correctBias(thisBPR->biasSensitivity, bias);
    }
    /* end of ABackpropNet interface */
};

/** A feed-forward neural network is a stack of layers. */
class Net : public ABackpropNet {
protected:
    std::vector<std::shared_ptr<ABackpropLayer>> layers;
public:
    Net() : layers(0) {}

    virtual Net &append(std::shared_ptr<ABackpropLayer> layer) {
        layers.push_back(std::move(layer));
        if (layers.size() >= 2) { // connect the last two layers
           auto n = layers.size();
           layers[n-2]->connectTo(*layers[n-1]);
        }
        return *this;
    }

    virtual void
    init(std::unique_ptr<RNG> &rng, WeightInit init = normalXavier) {
        for (size_t i = 0u; i < layers.size(); ++i) {
            layers[i]->init(rng, init);
        }
    }

    virtual void clear() { layers.clear(); }

    /* begin ABackpropNet interface */
    virtual std::shared_ptr<Array> output(const Array &inputs) {
        if (layers.empty()) {
            throw std::logic_error("no layers");
        }
        std::shared_ptr<Array> out = layers[0]->output(inputs);
        for (auto i = 1u; i < layers.size(); ++i) {
            out = layers[i]->output(*out);
        }
        return out;
    }

    virtual std::shared_ptr<BackpropResult> backprop(const Array &errors) {
        if (layers.empty()) {
            throw std::logic_error("no layers");
        }
        size_t n = layers.size();
        std::shared_ptr<BackpropResult> bpr;
        bpr = layers[n - 1]->backprop(errors);
        for (size_t offset = 1; offset < n; ++offset) {
            size_t i = n - 1 - offset;
            Array &e(bpr->propagatedErrors);
            bpr = layers[i]->backprop(e);
        }
        return bpr;
    }

    virtual void setLearningPolicy(const ALearningPolicy &lp) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->setLearningPolicy(lp);
        }
     }

    virtual void update() {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update();
        }
    }
    /* end ABackpropNet interface */

    using LayerIterator = decltype(layers.cbegin());
    LayerIterator begin() const { return layers.cbegin(); }
    LayerIterator end() const { return layers.cend(); }
};

/// Multiple `FullyConnectedLayer's stacked one upon another.
class MultilayerPerceptron : public Net {
public:
    MultilayerPerceptron() {}

    MultilayerPerceptron(std::initializer_list<unsigned int> shape,
                         const ActivationFunction &af = scaledTanh) {
        if (shape.size() <= 0) {
            throw std::invalid_argument("initializer list is empty");
        }
        auto pIn = shape.begin();
        auto pOut = std::next(pIn);
        for (; pOut != shape.end(); ++pIn, ++pOut) {
            auto inSize = *pIn;
            auto outSize = *pOut;
            std::shared_ptr<ABackpropLayer>
                layer(new FullyConnectedLayer(inSize, outSize, af));
            append(layer);
        }
    }

    /*
    // TODO: change signature and update when ALayer has .clone() method
    MultilayerPerceptron &append(const FullyConnectedLayer &layer) {
        std::shared_ptr<ABackpropLayer> new_layer(new FullyConnectedLayer(layer));
        return this->append(new_layer);
    }
    */
};

// TODO: CNN layer
// TODO: max-pooling layer
// TODO: decouple activation function from layer
// TODO: NN builder which takes Ciresan's string-like specs: 100c5-mp2-...
// TODO: NN formatters
// TODO: sliding window search for CNNs

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

// TODO: remove loss argument, use the top loss layer or L2 loss by default
/** Calculate total loss across the entire testSet. */
float totalLoss(LossFunction loss,
                ABackpropNet &net,
                const LabeledDataset& testSet) {
    float totalLoss = 0.0;
    for (auto sample : testSet) {
        auto out = net.output(sample.data);
        totalLoss += loss(*out, sample.label);
    }
    return totalLoss;
}

// return true from TrainCallback to stop training early
using TrainCallback = std::function<bool(int epoch)>;

/** Traing using stochastic gradient descent.
 *
 * Use net.setLearningPolicy() to change learning parameters of the network
 * _before_ calling `trainWithSGD`.
 *
 * @param net             Neural network to be trained.
 * @param trainSet        Training set.
 * @param rng             Random number generator for shuffling.
 * @param epochs          How many iterations (epochs) to run;
 *                        the entire training set is processed once per epoch.
 * @param callbackPeriod  A period of callback invocation if not zero;
 *                        The callback is also called before the first
 *                        and after the last iteration.
 * @param callback        A callback function to be invoked;
 *                        the callback may return true to stop training early.
 *
 * See:
 *
 *  - Efficient BackProp (2012) LeCun et al
 *    http://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf
 */
void trainWithSGD(ABackpropNet &net, LabeledDataset &trainSet,
        std::unique_ptr<RNG> &rng, int epochs,
        int callbackPeriod=0, TrainCallback callback=nullptr) {
    for (int j = 0; j < epochs; ++j) {
        if (callback && callbackPeriod > 0 && j % callbackPeriod == 0) {
            bool shouldStop = callback(j);
            if (shouldStop) {
                return;
            }
        }
        trainSet.shuffle(rng);
        for (auto sample : trainSet) {
            auto out_ptr = net.output(sample.data);
            // TODO: pass $d(E)/d(o_i)$ where $E$ is any loss function
            Array err = sample.label - *out_ptr; net.backprop(err);
            net.update();
        }
    }
    if (callback && callbackPeriod > 0) {
        callback(epochs);
    }
}

// TODO: softmax layer
// TODO: cross-entropy loss
// TODO: Hinge loss

#endif /* NOTCH_H */
