#ifndef NOTCH_H
#define NOTCH_H

// TODO: doxygen-compatible comments
// TODO: benchmarks

/// @file notch.hpp The main header file of the Notch neural networks library

/*

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

#include <algorithm>  // generate, transform
#include <array>      // array
#include <cmath>      // sqrt, exp
#include <functional> // ref, function<>
#include <initializer_list>
#include <iterator>   // begin, end
#include <memory>     // unique_ptr, make_unique
#include <numeric>    // inner_product
#include <random>
#include <string>
#include <sstream>    // ostringstream
#include <tuple>      // tuple, make_tuple
#include <typeinfo>   // typeid
#include <type_traits> // enable_if, is_pointer
#include <valarray>
#include <vector>

#ifdef NOTCH_USE_CBLAS
#include <cblas.h>
#endif

#ifdef NOTCH_USE_OPENMP
#include <omp.h>
#endif


#include "notch_debug.hpp"  // TODO: remove from release


namespace notch {

/* Library Framework
 * =================
 */

/* Random number generation
 * ------------------------
 */

/// A synonym for the random number generator engine type.
using RNG = std::mt19937;

/* Data types
 * ----------
 *
 * A neural network consumes a vector of numerical values, and produces a vector
 * of numerical outputs. Without too much loss of generality we may consider
 * them arrays of single-precision floating point numbers.
 */

/** Library array type.
 *
 * The library uses C++ std::valarray to store network `Input`, `Output` and
 * parameters to make code more concise and expressive (valarrays implement
 * elementwise operations and slices). */
using Array = std::valarray<float>;
using Input = Array;
using Output = Array;

/** Unlabeled dataset is a collection or `Input`s or `Output`s. */
using Dataset = std::vector<Array>;


/** A common interface to preprocess data.
 *
 *  Transformations are supposed to be invertable. */
class ADatasetTransformer {
public:
    virtual Dataset apply(const Dataset &) = 0;
    virtual Array apply(const Array &) = 0;
    virtual Dataset unapply(const Dataset &) = 0;
    virtual Array unapply(const Array &) = 0;
};


/** A pair of data vector and its label for supervised learning.
 *
 * Supervised learning requires labeled data.
 * A label is a vector of numeric values. */
struct LabeledData {
    Input const &data;
    Output const &label;
};


/** Multiple `LabeledData` samples.
 *
 * `LabeledDataset`s can be used like training or testing sets. */
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
    void apply(ADatasetTransformer &t) {
        inputs = t.apply(inputs);
        if (!inputs.empty()) {
            inputDimension = inputs[0].size();
        }
    }

    /// Preprocess `Output` labels
    void applyToLabels(ADatasetTransformer &t) {
        outputs = t.apply(outputs);
        if (!outputs.empty()) {
            outputDimension = outputs[0].size();
        }
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

    /// Keep only the first `newSize` samples. Discard everything else.
    void truncate(size_t newSize) {
        if (newSize < size()) {
            inputs.resize(newSize);
            outputs.resize(newSize);
            nSamples = inputs.size();
        }
    }
};


namespace internal {

/* Linear Algebra
 * ==============
 */

#ifdef NOTCH_USE_CBLAS

/** Matrix-vector product, similar to BLAS _gemv function.
 * Calculate $b = \mathbf{M}*x + b$.
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
        std::ostringstream what;
        what << "blas_gemv: incompatible matrix and vector shapes:\n"
            << " matrix size = " << n
            << " vector size = " << cols
            << " result size = " << rows;
        throw std::invalid_argument(what.str());
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                rows, cols,
                1.0 /* M multiplier */,
                m_begin, cols /* leading dimension of M */,
                x_begin, 1,
                1.0 /* b multiplier */,
                b_begin, 1);
}

/** Vector-vector dot product, similar to BLAS _dot function.
 *
 * See gemv notes. */
template <class VectorX_Iter, class VectorY_Iter>
typename std::enable_if<std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorY_Iter>::value,
float>::type
dot(VectorX_Iter x_begin, VectorX_Iter x_end,
    VectorY_Iter y_begin, VectorY_Iter y_end) {
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "blas_dot: vector sizes don't match:\n"
            << " x size = " << x_size
            << " y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
    return cblas_sdot(x_size, x_begin, 1, y_begin, 1);
}

/** Vector-vector dot product with strides, similar to BLAS _dot function.
 *
 * See gemv notes. */
template <class VectorX_Iter, class VectorY_Iter>
typename std::enable_if<std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorY_Iter>::value,
float>::type
dot(const size_t n,
    VectorX_Iter x_begin, const size_t x_stride,
    VectorY_Iter y_begin, const size_t y_stride) {
    return cblas_sdot(n, x_begin, x_stride, y_begin, y_stride);
}

/** Outer product between two vectors. Calculate $\mathbf{M} = \alpha x y^T$.
 *
 * See gemv notes. */
template <class VectorX_Iter, class VectorY_Iter, class Matrix_OutIter>
typename std::enable_if<std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorY_Iter>::value &&
                        std::is_pointer<Matrix_OutIter>::value, void>::type
outer(float alpha,
      VectorX_Iter x_begin, VectorX_Iter x_end,
      VectorY_Iter y_begin, VectorY_Iter y_end,
      Matrix_OutIter m_begin, Matrix_OutIter m_end) {
    size_t rows = std::distance(x_begin, x_end);
    size_t cols = std::distance(y_begin, y_end);
    size_t n = std::distance(m_begin, m_end);
    if (n != rows * cols) {
        std::ostringstream what;
        what << "cblas_outer: incompatible shapes:\n"
            << " vector X size = " << rows
            << " vector Y size = " << cols
            << " result size = " << n;
        throw std::invalid_argument(what.str());
    }
    std::fill(m_begin, m_end, 0.0);
    cblas_sger(CblasRowMajor, rows, cols,
               alpha, x_begin, 1,
               y_begin, 1,
               m_begin, cols /* leading dimension of M */);
}

/** Multiply a vector by a scalar: Calculate $y = \alpha x$.
 *
 * See gemv notes. */
template <class VectorX_Iter, class VectorY_OutIter>
typename std::enable_if<std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorY_OutIter>::value,
void>::type
scale(float alpha,
      VectorX_Iter x_begin, VectorX_Iter x_end,
      VectorY_OutIter y_begin, VectorY_OutIter y_end) {
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "cblas_scale: incompatible shapes:\n"
            << " vector X size = " << x_size
            << " vector Y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
    std::fill(y_begin, y_end, 0.0);
    cblas_saxpy(x_size, alpha, x_begin, 1, y_begin, 1);
}

/** Multiply a vector by a scalar and add to another vector,
 * similar to BLAS _axpy function. Calculate $y = \alpha x + y$.
 *
 * See gemv notes. */
template <class VectorX_Iter, class VectorY_OutIter>
typename std::enable_if<std::is_pointer<VectorX_Iter>::value &&
                        std::is_pointer<VectorY_OutIter>::value,
void>::type
scaleAdd(float alpha,
         VectorX_Iter x_begin, VectorX_Iter x_end,
         VectorY_OutIter y_begin, VectorY_OutIter y_end) {
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "cblas_scaleAdd: incompatible shapes:\n"
            << " vector X size = " << x_size
            << " vector Y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
    cblas_saxpy(x_size, alpha, x_begin, 1, y_begin, 1);
}

#else /* NOTCH_USE_CBLAS is not defined */

/** Matrix-vector product, similar to BLAS _gemv function.
 * Calculate $b = \mathbf{M}*x + b$. */
template <class Matrix_Iter, class VectorX_Iter, class VectorB_Iter>
void
gemv(Matrix_Iter m_begin, Matrix_Iter m_end,
     VectorX_Iter x_begin, VectorX_Iter x_end,
     VectorB_Iter b_begin, VectorB_Iter b_end) {
    size_t cols = std::distance(x_begin, x_end);
    size_t rows = std::distance(b_begin, b_end);
    size_t n = std::distance(m_begin, m_end);
    if (n != rows * cols) {
        std::ostringstream what;
        what << "stl_gemv: incompatible matrix and vector shapes:\n"
            << " matrix size = " << n
            << " vector size = " << cols
            << " result size = " << rows;
        throw std::invalid_argument(what.str());
    }
    size_t r = 0; // current row number
    for (auto b = b_begin; b != b_end; ++b, ++r) {
        auto row_begin = m_begin + r * cols;
        auto row_end = row_begin + cols;
        *b = std::inner_product(row_begin, row_end, x_begin, *b);
    }
}

/** Vector-vector dot product, similar to BLAS _dot function. */
template <class VectorX_Iter, class VectorY_Iter>
float
dot(VectorX_Iter x_begin, VectorX_Iter x_end,
    VectorY_Iter y_begin, VectorY_Iter y_end) {
    double dotProduct = 0.0;
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "stl_dot: vector sizes don't match:\n"
            << " x size = " << x_size
            << " y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
    auto x = x_begin;
    auto y = y_begin;
    for (; x != x_end; ++x, ++y) {
        dotProduct += (*x) * (*y);
    }
    return static_cast<float>(dotProduct);
}

/** Vector-vector dot product with strides, similar to BLAS _dot function. */
template <class VectorX_Iter, class VectorY_Iter>
float
dot(const size_t n,
    VectorX_Iter x_begin, const size_t x_stride,
    VectorY_Iter y_begin, const size_t y_stride) {
    double dotProduct = 0.0;
    auto x = x_begin;
    auto y = y_begin;
    for (size_t i = 0; i < n; ++i) {
        dotProduct += (*x) * (*y);
        x += x_stride;
        y += y_stride;
    }
    return static_cast<float>(dotProduct);
}

/** Outer product between two vectors. Calculate $\mathbf{M} = \alpha x y^T$. */
template <class VectorX_Iter, class VectorY_Iter, class Matrix_OutIter>
void
outer(float alpha,
      VectorX_Iter x_begin, VectorX_Iter x_end,
      VectorY_Iter y_begin, VectorY_Iter y_end,
      Matrix_OutIter m_begin, Matrix_OutIter m_end) {
    size_t rows = std::distance(x_begin, x_end);
    size_t cols = std::distance(y_begin, y_end);
    size_t n = std::distance(m_begin, m_end);
    if (n != rows * cols) {
        std::ostringstream what;
        what << "stl_outer: incompatible shapes:\n"
            << " vector X size = " << rows
            << " vector Y size = " << cols
            << " result size = " << n;
        throw std::invalid_argument(what.str());
    }
#ifdef NOTCH_USE_OPENMP
#pragma omp parallel for shared(m_begin, x_begin, y_begin, rows, cols, alpha)
#endif
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            float &m_rc = *(m_begin + r*cols + c);
            float x_r = *(x_begin + r);
            float y_c = *(y_begin + c);
            m_rc = alpha * x_r * y_c;
        }
    }
}

/** Multiply vector by a scalar: Calculate $y = \alpha x$. */
template <class VectorX_Iter, class VectorY_OutIter>
void
scale(float alpha,
      VectorX_Iter x_begin, VectorX_Iter x_end,
      VectorY_OutIter y_begin, VectorY_OutIter y_end) {
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "stl_scale: incompatible shapes:\n"
            << " vector X size = " << x_size
            << " vector Y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
#ifdef NOTCH_USE_OPENMP
#pragma omp parallel for shared(x_begin, y_begin, x_size, alpha)
#endif
    for (size_t i = 0; i < x_size; ++i) {
        float x_i = *(x_begin + i);
        float &y_i = *(y_begin + i);
        y_i = alpha * x_i;
    }
}

/** Multiply a vector by a scalar and add to another vector,
 * similar to BLAS _axpy function.  Calculate $y = \alpha x + y$. */
template <class VectorX_Iter, class VectorY_OutIter>
void
scaleAdd(float alpha,
         VectorX_Iter x_begin, VectorX_Iter x_end,
         VectorY_OutIter y_begin, VectorY_OutIter y_end) {
    size_t x_size = std::distance(x_begin, x_end);
    size_t y_size = std::distance(y_begin, y_end);
    if (x_size != y_size) {
        std::ostringstream what;
        what << "stl_scaleAdd: incompatible shapes:\n"
            << " vector X size = " << x_size
            << " vector Y size = " << y_size;
        throw std::invalid_argument(what.str());
    }
#ifdef NOTCH_USE_OPENMP
#pragma omp parallel for shared(x_begin, y_begin, x_size, alpha)
#endif
    for (size_t i = 0; i < x_size; ++i) {
        float x_i = *(x_begin + i);
        float &y_i = *(y_begin + i);
        y_i += alpha * x_i;
    }
}

#endif /* ifdef NOTCH_USE_CBLAS */

} // end of namespace notch::internal

/* Neurons and Neural Networks
 * ===========================
 */


/* Random Weights Initialization
 * -----------------------------
 */


using WeightInit =
    std::function<void(std::unique_ptr<RNG> &rng,
                       Array &weights, int nInputs, int nOutputs)>;


class Init {
public:
    /// Create and seed a new random number generator.
    static std::unique_ptr<RNG> newRNG() {
        std::random_device rd;
        std::array<uint32_t, std::mt19937::state_size> seed_data;
        std::generate(seed_data.begin(), seed_data.end(), ref(rd));
        std::seed_seq sseq(std::begin(seed_data), std::end(seed_data));
        std::unique_ptr<RNG> rng(new RNG());
        rng->seed(sseq);
        return rng;
    }

    /** Uniform Nguyen-Widrow initialization.
     *
     * It is assumed that elements of the input vector are in the range [-1,+1].
     * The magnitude of weight vectors is adjusted randomly so that each node
     * is linear over only a small interval.
     *
     * References:
     *
     *  - Nguyen; Widrow (1990) Improving the Learning Speed of 2-Layer
     *    Neural Networks by Choosing Initial Values of the Adaptive Weights
     *  - http://dsp.vscht.cz/konference_matlab/matlab04/pavelka.pdf
     *  - http://www.heatonresearch.com/encog/articles/nguyen-widrow-neural-network-weight.html
     **/
    static
    void uniformNguyenWidrow(std::unique_ptr<RNG> &rng,
                             Array &weights, int nIn, int nOut) {
        std::uniform_real_distribution<float> U(-1, +1);
        float overlap = 0.7; // "to have the intervals overlap slightly"
        float beta = overlap * pow(nOut, 1.0 / nIn); // target row norm
        if (weights.size() == static_cast<size_t>(nIn*nOut)) {
            // we're initializing weights
            float w_row_norm = sqrt(nIn / 3.0); // for nIn elems in U([-1,1])
            std::generate(std::begin(weights), std::end(weights),
                    [&U, &rng, beta, w_row_norm] {
                        // w_{ji} in U([-1,1])
                        float w_ji = U(*rng.get());
                        // to have row norm equal beta
                        return w_ji * beta / w_row_norm;
                    });
        } else { // we're initializing bias
            // b_i = uniform random number in [-beta, beta]
            std::generate(std::begin(weights), std::end(weights),
                    [&U, &rng, beta] {
                        float u = U(*rng.get());
                        return u * beta;
                    });
         }
    }

    /** One-sided Xavier or "Fan-in" initialization.
     *
     * Pick weights from a zero-centered _normal_ distrubition with variance
     * $$\sigma^2 = 1/n_{in}$$, where $n_{in}$ is the number of inputs.
     *
     * See
     *
     *  - NNLM3, Chapter 4, page 149;
     *  - Glorot, Xavier; Bengio, Yoshua. (2010). Understanding the difficuly
     *    of training deep feedforward neural networks.
     *  - http://andyljones.tumblr.com/post/110998971763/
     **/
    static
    void normalXavier(std::unique_ptr<RNG> &rng, Array &weights, int n_in, int) {
        float sigma = n_in > 0 ? sqrt(1.0 / n_in) : 1.0;
        std::normal_distribution<float> nd(0.0, sigma);
        std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
            float w = nd(*rng.get());
            return w;
        });
    }

    /** Uniform one-sided Xavier or "Fan-in" initialization.
     *
     * Pick weights from a zero-centered _uniform_ distrubition with variance
     * $$\sigma^2 = 1/n_{in}$$, where $n_{in}$ is the number of inputs.
     *
     * See
     *
     *  - NNLM3, Chapter 4, page 149;
     *  - Glorot, Xavier; Bengio, Yoshua. (2010). Understanding the difficuly
     *    of training deep feedforward neural networks.
     *  - http://andyljones.tumblr.com/post/110998971763/
     **/
    static
    void uniformXavier(std::unique_ptr<RNG> &rng, Array &weights, int n_in, int) {
        float sigma = n_in > 0 ? sqrt(1.0/n_in) : 1.0;
        float a = sigma * sqrt(3.0);
        std::uniform_real_distribution<float> nd(-a, a);
        std::generate(std::begin(weights), std::end(weights), [&nd, &rng] {
                    float w = nd(*rng.get());
                    return w;
                 });
    }
};


/* Activation Functions
 * --------------------
 */

 /** Activation functions are applied to neuron's output to introduce
 * non-linearity and map output to specific range. Backpropagation algorithm
 * requires differentiable activation functions. */
class Activation {
public:
    virtual std::string tag() const = 0;
    virtual float operator()(float v) const = 0;
    virtual float derivative(float v) const = 0;
};


/** Logistic function maps output to the range (0,1).
 *
 * It is defined as
 *
 * $$\phi(v) = 1/(1 + \exp(- s v)),$$
 *
 * where $s$ is its derivative at zero.
 * See  NNLM3, Chapter 4, page 135. */
class LogisticActivation : public Activation {
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

    virtual std::string tag() const {
        std::ostringstream out;
        out << "logistic";
        if (slope != 1.0) {
            out << "(" << slope << ")";
        }
        return out.str();
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
 * - Y. LeCun et al. (2012) Efficient BackProp. In: NNTT.
 */
class TanhActivation : public Activation {
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

    virtual std::string tag() const { return name; }
};


/** Piecewise linear activation function allows to implement
 * rectified linear units (ReLU) and leaky rectified linear units (leakyReLU).
 *
 * Rectifiers are often used in deep neural networks because they don't
 * suffer from diminishing gradients problem and allow for sparse activation
 * (neurons with negative induced local field remain dormant). */
class PiecewiseLinearActivation : public Activation {
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

    virtual std::string tag() const { return name; }
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


/* Multilayer Perceptrons
 * ----------------------
 */

/** A base class for the rule to correct weights and bias given sensitivity factors.
 *
 * Concrete implementations may overwrite weightSensitivity and biasSensitivity
 * parameters. Output is written to weighs and bias parameters. */
class ALearningPolicy {
public:
    virtual void update(Array &weights, Array &bias,
                        Array &weightSensitivity, Array &biasSensitivity) = 0;
    virtual std::unique_ptr<ALearningPolicy> clone() const = 0;
};


/** Learning with fixed rate, momentum term and weight-decay.
 *
 * Generalized delta rule for learning with fixed rate with a momentum term:
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
 * If the momentum $\alpha = 0$ then the learning policy becomes classic delta
 * rule:
 *
 * $$\Delta w_{ji} = - \eta \partial E / \partial w_{ji}.$$
 *
 * If weightDecay parameter $\lambda$ is not zero, it has an effect of
 * adding a complexity penalty term to the loss function:
 *
 * $$E_c(\mathbf{w}) = \lambda ||\mathbf{w}||^2$$
 *
 * and accordingly an extra term to $\partial E/\partial w_{ji}$:
 *
 * $$\partial E_c/\partial w_{ji} = 2 \lambda \mathbf{w}_{ji}$$
 *
 * Weight-decay has an effect of reducing excess weights of the network
 * (weights which have little influence on the network performance).
 * It may improve generatlization of the network.
 * For a method how to choose parameter $\lambda$ see (Rognvaldsson, 2012).
 *
 * References:
 *
 *  - Delta rule: NNLM3, Eq (4.14), page 131
 *  - Generalized delta rule: NNLM3, Eq (4.41), page 137
 *  - LeCun (2012) Efficient Backprop, page 12 and 21. In: NNTT
 *  - NNLM3, Weight-decay procedure, Eq (4.96), page 176.
 *  - Rognvaldsson (2012) A Simple Trick ..., page 71. In: NNTT
 **/
class FixedRate : public ALearningPolicy {
private:
    float learningRate;
    float momentum;
    float weightDecay;
    Array lastDeltaW;
    Array lastDeltaB;

    /// update @param var using a classic delta rule
    void deltaRule(Array& var, const Array &grad) {
        size_t n = var.size();
        for (size_t i = 0; i < n; ++i) {
            float &var_i = var[i];
            float delta_i = - learningRate * grad[i];
            if (weightDecay != 0.0) {
                delta_i -= learningRate * 2.0 * weightDecay * var_i;
            }
            var_i += delta_i;
        }
    }

    /// udpate @param var using a generalized delta rule
    void deltaRuleWithMomentum(Array &var, Array &lastDelta, const Array &grad) {
        size_t n = var.size();
        if (lastDelta.size() != n) {
            lastDelta.resize(n, 0.0);
        }
#ifdef NOTCH_DISABLE_OPTIMIZATIONS /* original version */
        for (size_t i = 0; i < n; ++i) {
            float &lastDelta_i = lastDelta[i];
            float &var_i = var[i];
            float grad_i = grad[i];
            float delta_i = momentum * lastDelta_i - learningRate * grad_i;
            if (weightDecay != 0.0) {
                delta_i -= learningRate * 2.0 * weightDecay * var_i;
            }
            lastDelta_i = delta_i;
            var_i += delta_i;
        }
#else /* optimized version (fewer operator[] calls) */
        auto last_delta_ptr = std::begin(lastDelta);
        auto grad_ptr = std::begin(grad);
        auto var_ptr = std::begin(var);
        for (size_t i = 0; i < n; ++i) {
            float last_delta_i = (*last_delta_ptr);
            float grad_i = (*grad_ptr);
            float delta_i = momentum * last_delta_i - learningRate * grad_i;
            if (weightDecay != 0.0) {
                delta_i -= learningRate * 2.0 * weightDecay * (*var_ptr);
            }
            (*last_delta_ptr) = delta_i;
            (*var_ptr) += delta_i;
            ++last_delta_ptr; ++grad_ptr; ++var_ptr;
        }
#endif
    }

public:
    FixedRate(float learningRate = 0.01,
              float momentum = 0.0,
              float weightDecay = 0.0)
        : learningRate(learningRate),
          momentum(momentum),
          weightDecay(weightDecay) {}

    virtual void update(Array &weights, Array &bias,
                        Array &weightSensitivity, Array &biasSensitivity) {
        if (momentum == 0.0) {
            deltaRule(weights, weightSensitivity);
            deltaRule(bias, biasSensitivity);
        } else {
            deltaRuleWithMomentum(weights, lastDeltaW, weightSensitivity);
            deltaRuleWithMomentum(bias, lastDeltaB, biasSensitivity);
        }
    }

    virtual std::unique_ptr<ALearningPolicy> clone() const {
        float lr = learningRate;
        float m = momentum;
        float wd = weightDecay;
        auto c = std::unique_ptr<ALearningPolicy>(new FixedRate(lr, m, wd));
        return c;
    }
};



/** ADADELTA, an adaptive learning rate method.
 *
 * The method is based on two (and a half) ideas:
 *
 *  - make weight updates inversely proportional to its root of the
 *    mean squared gradient over a finite time window
 *    (i.e. reduce the effective learning rate if the error gradient
 *    grows, increase the effective learning if the gradient dissipates)
 *
 *  - the dimensional units of the weight updates should be the
 *    same as weight units; so the weight updates are made proportional
 *    to the mean squared of the previous updates
 *
 *  - (the half of an idea): regularize both RMSs in such a way that
 *    the progress continues even when gradient and recent weight updates
 *    tend to zero.
 *
 * The difference with respect to (Zeiler, 2012): the rate of weight
 * and bias updates can be different.
 *
 * References:
 *
 *  - Zeiler (2012) ADADELTA: An Adaptive Learning Rate Method
 */
class AdaDelta : public ALearningPolicy {
protected:
    float momentum; //< $\rho$, exponential smoothing parameter in (Zeiler, 2012)
    float epsilon; // $\epsilon$ regularization parameter in (Zeiler, 2012)
    float squaredWeightsDelta = 0.0;
    float squaredBiasDelta = 0.0;
    float squaredGrad_w = 0.0;
    float squaredGrad_b = 0.0;
    // TODO: check if using the same rate for weight and bias (Zeiler, 2012) is better

    void adaDeltaRule(Array &var, const Array &grad,
                     float &squaredVarDelta, float &squaredGradAccum) {
        auto g_begin = std::begin(grad);
        auto g_end = std::end(grad);
        // compute gradient norm
        float grad2 = internal::dot(g_begin, g_end, g_begin, g_end);
        // accumuate squared gradient (with exponential smoothing)
        squaredGradAccum = momentum*squaredGradAccum + (1-momentum)*grad2;
        // compute updates (time step)
        float grad_RMS = sqrt(squaredGradAccum + epsilon);
        float delta_RMS = sqrt(squaredVarDelta + epsilon);
        float eta = delta_RMS / grad_RMS;
        // accumuate squared updates (with exponential smoothing)
        squaredVarDelta = momentum*squaredVarDelta + (1-momentum)*grad2*eta*eta;
        // apply updates
#ifdef NOTCH_DISABLE_OPTIMIZATIONS /* original version */
        var = var - (eta * grad);
#else /* optimized version */
        size_t n = var.size();
        // TODO: internal::scaleAdd(-eta, n, std::begin(grad), std::begin(var));
        auto grad_ptr = std::begin(grad);
        auto var_ptr = std::begin(var);
#ifdef NOTCH_USE_OPENMP
#pragma omp parallel for shared(var_ptr, grad_ptr)
#endif
        for (size_t i = 0; i < n; ++i) {
            float grad_i = (*(grad_ptr + i));
            float delta_i = - eta * grad_i;
            float &var_i = (*(var_ptr + i));
            var_i += delta_i;
        }
#endif
    }

public:
    AdaDelta(float momentum=0.95, float epsilon=1e-6)
        : momentum(momentum), epsilon(epsilon) {}

    virtual void update(Array &weights, Array &bias,
                        Array &weightSensitivity, Array &biasSensitivity) {
        adaDeltaRule(weights, weightSensitivity,
                     squaredWeightsDelta, squaredGrad_w);
        adaDeltaRule(bias, biasSensitivity,
                     squaredBiasDelta, squaredGrad_b);
    }

    virtual std::unique_ptr<ALearningPolicy> clone() const {
        float m = momentum;
        float e = epsilon;
        auto c = std::unique_ptr<ALearningPolicy>(new AdaDelta(m, e));
        return c;
    }
};


/** Input-output buffers can be shared between layers.
 *
 * SharedBuffers are allocated on-demand only when layers
 * are actually connected. This allows to avoid allocating
 * the same buffer twice. */
class SharedBuffers {
public:
    std::shared_ptr<Array> inputBuffer;
    std::shared_ptr<Array> outputBuffer;
    bool ready() const {
        return inputBuffer && outputBuffer;
    }
    /// allocate input-output buffers
    void allocate(size_t nInputs, size_t nOutputs) {
        if (ready()) {
            return;
        }
        if (!inputBuffer) {
            inputBuffer = std::make_shared<Array>(0.0, nInputs);
        }
        if (!outputBuffer) {
            outputBuffer = std::make_shared<Array>(0.0, nOutputs);
        }
    }
   /// clone buffers (make them non-shared)
    SharedBuffers clone() const {
        SharedBuffers newBuffers;
        if (inputBuffer) {
            newBuffers.inputBuffer = std::make_shared<Array>(*inputBuffer);
        }
        if (outputBuffer) {
            newBuffers.outputBuffer = std::make_shared<Array>(*outputBuffer);
        }
        return newBuffers;
    }
};


/** An intermediate layer of the network with back-propagation capability. */
class ABackpropLayer {
protected:
    // Forward and backpropagation results can be shared between layers.
    // The buffers are allocated once either in connect() or at the begining of
    // the forward or backprop pass respectively (see output() and backprop()).
    SharedBuffers shared;

public:
    /// A name to identify layer type.
    virtual std::string tag() const = 0;

    /// Get the number of input variables.
    virtual size_t inputDim() const = 0;
    /// Get the number of output variables.
    virtual size_t outputDim() const = 0;

    /// Randomly initialize layer parameters.
    virtual void init(std::unique_ptr<RNG> &rng, WeightInit init) = 0;

    /// Get a pointer to the last input buffer which can be shared with other layers.
    virtual std::shared_ptr<Array> getInputBuffer() = 0;
    /// Get a pointer to the last output buffer which can be shared with other layers.
    virtual std::shared_ptr<Array> getOutputBuffer() = 0;

    /// Get a pointer to weight parameters. It can be nullptr if the layer has no parameters.
    virtual std::shared_ptr<const Array> getWeights() const { return nullptr; }
    /// Get a pointer to bias parameters. It can be nullptr if the layer has no parameters.
    virtual std::shared_ptr<const Array> getBias() const { return nullptr; }
    /// Get a pointer to last calculated weight sensitivity (gradient of the network loss
    /// with respect to the layer weight parameters). It can be nullptr.
    virtual std::shared_ptr<const Array> getWeightSensitivity() const { return nullptr; }
    /// Get a pointer to last calculated bias sensitivity (gradient of the network loss
    /// with respect to the layer bias parameters). It can be nullptr.
    virtual std::shared_ptr<const Array> getBiasSensitivity() const { return nullptr; }
    /// Get layer activation function.
    virtual const Activation &getActivation() const { return linearActivation; }

    /// Create a copy of the layer with its own detached buffers.
    virtual std::shared_ptr<ABackpropLayer> clone() const = 0;

    /// Forward propagaiton pass.
    ///
    /// @return inputs for the next layer.
    virtual const Array &output(const Array &inputs) = 0;
    /// Back propagation pass.
    ///
    /// @return error signals $e_i$ propagated to the previous layer.
    virtual const Array &backprop(const Array &errors) = 0;
    /// Specify how layer parameters have to be adjusted.
    virtual void setLearningPolicy(const ALearningPolicy &lp) = 0;
    /// Adjust layer parameters after the backpropagation pass.
    virtual void update() = 0;
};


/** The top-most layer of the network.
 *
 * Loss layers compare calculated output of the network with the desired
 * output, calculate loss and error gradient to propagate back. */
class ALossLayer {
protected:
    SharedBuffers shared;

public:
    /// A name to identify layer type.
    virtual std::string tag() const { return "ALossLayer"; }

    /// Calculate loss.
    virtual float output(const Array &actual, const Array &expected) = 0;

    /// Return error signals $e_i$ to propagate to the previous layer.
    virtual const Array &backprop() = 0;

    /// Create a copy of the layer with its own detached buffers.
    virtual std::shared_ptr<ALossLayer> clone() const = 0;

    virtual size_t inputDim() const = 0;
    virtual size_t outputDim() const { return 0; }
};


template<class LAYER>
class GetShared : public LAYER {
public:
    /// Get a reference to protected 'shared' member of a LAYER class.
    static SharedBuffers& ref(LAYER &l) {
        auto &access = static_cast<GetShared<LAYER>&>(l);
        return access.shared;
    }
};


/** connect<PREV_LAYER, NEXT_LAYER>() allows to connect layers
 * of different types to each other.
 *
 * Both layer classes are expected to have these members:
 *
 *  - 'protected SharedBuffers shared'
 *  - inputDim() and outputDim() methods
 *  - getInputBuffer() and getOutputBuffer() methods
 */
template<class PREV_LAYER, class NEXT_LAYER>
void connect(PREV_LAYER &prevLayer, NEXT_LAYER &nextLayer) {
    size_t prevIn = prevLayer.inputDim();
    size_t prevOut = prevLayer.outputDim();
    size_t nextIn = nextLayer.inputDim();
    size_t nextOut = nextLayer.outputDim();
    if (prevOut != nextIn) {
        std::ostringstream ss;
        ss << "can't connect " << prevLayer.tag() << " "
           << prevIn << " in/" << prevOut << " out"
           << " to " << nextLayer.tag() << " "
           << nextIn << " in/" << nextOut << " out";
        throw std::invalid_argument(ss.str());
    }
    GetShared<PREV_LAYER>::ref(prevLayer).allocate(prevIn, prevOut);
    GetShared<NEXT_LAYER>::ref(nextLayer).inputBuffer = prevLayer.getOutputBuffer();
}


/** A fully connected layer of neurons with backpropagation. */
class FullyConnectedLayer : public ABackpropLayer {
protected:
    size_t nInputs;
    size_t nOutputs; //< the number of neurons in the layer
    std::shared_ptr<Array> weights; //< weights matrix $w_ji$ for the entire layer, row-major order
    std::shared_ptr<Array> bias;    //< bias values $b_j$ for the entire layer, one per neuron
    const Activation *activationFunction;

    Array inducedLocalField;  //< $v_j = \sum_i w_ji x_i + b_j$
    Array activationGrad;     //< $\partial y/\partial v_j = \phi^\prime (v_j)$
    Array localGrad;          //< $\delta_j = \partial E/\partial v_j = \phi^\prime(v_j) e_j$
    std::shared_ptr<Array> weightSensitivity;  //< $\partial E/\partial w_{ji}$
    std::shared_ptr<Array> biasSensitivity;    //< $\partial E/\partial b_{j}$
    Array propagatedErrors; //< backpropagation result

    std::shared_ptr<ALearningPolicy> policy;

    std::shared_ptr<ABackpropLayer> makeClone() const {
        auto p = std::make_shared<FullyConnectedLayer>(*this);
        if (!p) {
            throw std::runtime_error("cannot clone layer");
        }
        p->shared = shared.clone();
        p->weights = std::make_shared<Array>(*weights);
        p->bias = std::make_shared<Array>(*bias);
        if (policy) {
            p->policy = policy->clone();
        }
        return p;
    }

    void rememberInputs(const Array& inputs) {
        *(shared.inputBuffer) = inputs;  // remember for calcSensitivityFactors()
    }

    /** Linear response of the layer $v_j = w_{ji} x_i$. */
    void calcInducedLocalField(const Array &inputs) {
        inducedLocalField = *bias; // will be added and overwritten
        internal::gemv(
             std::begin(*weights), std::end(*weights), std::begin(inputs),
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
    void calcOutput() {
        Array &outputs = *shared.outputBuffer;
        std::transform(std::begin(inducedLocalField),
                       std::end(inducedLocalField),
                       std::begin(outputs),
                       [&](float y) { return (*activationFunction)(y); });
    }

    void outputInplace(const Array &inputs) {
        Array &outputs = *shared.outputBuffer;
        if (outputs.size() != nOutputs) {
            outputs.resize(nOutputs);
        }
        rememberInputs(inputs);
        calcInducedLocalField(inputs);
        calcActivationGrad();
        calcOutput();
    }

    /** The local gradient $\delta_j = \partial E/\partial v_j$ is the product
     * of the activation function derivative and the error signal. */
    void calcLocalGrad(const Array &errors) {
        assert(activationGrad.size() == errors.size());
        assert(localGrad.size() == errors.size());
        localGrad = activationGrad * errors;
    }

    /** Calculate sensitivity factors $\\partial E/\\partial w_{ji}$ and
     * $\\partial E/\\partial b_j$ for weights and bias respectively.
     *
     * The derivative are then used to corrections to the matrix of the
     * synaptic weights and biases.
     *
     * NNLM3, Page 134, Eq. (4.27) defines weight correction as
     *
     * $$ \\Delta w_{ji} (n) = \\eta \\times \\delta_j (n) \\times y_{i} (n) $$
     *
     * where `w_{ji}` is the synaptic weight connecting neuron $i$ to neuron $j$,
     * `\eta` is learning rate, $delta_j (n)$ is the local [error] gradient
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
    void calcSensitivityFactors() {
        assert(weightSensitivity->size() == weights->size());
        assert(biasSensitivity->size() == bias->size());
        assert(localGrad.size() == nOutputs);
        Array &input = *shared.inputBuffer;
        assert(input.size() == nInputs);
        auto &dEdw = *weightSensitivity;
        auto &dEdb = *biasSensitivity;
        internal::outer(-1.0,                             // -
              std::begin(localGrad), std::end(localGrad), // \delta_j
              std::begin(input), std::end(input),         // y_i
              std::begin(dEdw), std::end(dEdw));          // =: dE/dw_{ji}
        internal::scale(-1.0,                             // -
              std::begin(localGrad), std::end(localGrad), // \delta_j
              std::begin(dEdb), std::end(dEdb));          // =: dE/db_j
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
    void calcPropagatedErrors() {
        if (propagatedErrors.size() != nInputs) {
            propagatedErrors.resize(nInputs);
        }
        // propagated errors are dot products between columns of the
        // weight matrix $w_{kj} and local gradient $\delta_j$;
        auto e_begin = std::begin(propagatedErrors); // points to e_0
        auto w_begin = std::begin(*weights);         // points to w_00
        auto delta_begin = std::begin(localGrad);    // points to delta_0
        for (size_t j = 0; j < nInputs; ++j) { // for all inputs
            float &e_j = *(e_begin + j);
            e_j = internal::dot(nOutputs, w_begin + j, nInputs, delta_begin, 1);
        }
    }

    /// Backpropagation algorithm
    void
    backpropInplace(const Array &errors) {
        calcLocalGrad(errors);
        calcSensitivityFactors();
        calcPropagatedErrors();
    }

    /// @return true if input-output buffers are allocated _and_ shared
    bool isConnected() const {
        return (shared.ready() &&
                !(shared.inputBuffer.unique() &&
                  shared.outputBuffer.unique()));
    }

public:
    /// Create a layer with zero weights.
    FullyConnectedLayer(size_t nInputs = 0, size_t nOutputs = 0,
                        const Activation &af = linearActivation)
        : nInputs(nInputs), nOutputs(nOutputs),
          weights(new Array(nInputs * nOutputs)), bias(new Array(nOutputs)),
          activationFunction(&af), inducedLocalField(nOutputs),
          activationGrad(nOutputs), localGrad(nOutputs),
          weightSensitivity(new Array(nInputs * nOutputs)),
          biasSensitivity(new Array(nOutputs)),
          propagatedErrors(nInputs) {}

    /// Create a layer from a weights matrix.
    FullyConnectedLayer(Array &&weights, Array &&bias,
                        const Activation &af)
        : nInputs(weights.size()/bias.size()), nOutputs(bias.size()),
          weights(new Array(weights)), bias(new Array(bias)),
          activationFunction(&af), inducedLocalField(nOutputs),
          activationGrad(nOutputs), localGrad(nOutputs),
          weightSensitivity(new Array(nInputs * nOutputs)),
          biasSensitivity(new Array(nOutputs)),
          propagatedErrors(nInputs) {}

    /// Create a layer from a copy of a weights matrix.
    FullyConnectedLayer(const Array &weights, const Array &bias,
                        const Activation &af)
        : nInputs(weights.size()/bias.size()), nOutputs(bias.size()),
          weights(new Array(weights)), bias(new Array(bias)),
          activationFunction(&af), inducedLocalField(nOutputs),
          activationGrad(nOutputs), localGrad(nOutputs),
          weightSensitivity(new Array(nInputs * nOutputs)),
          biasSensitivity(new Array(nOutputs)),
          propagatedErrors(nInputs) {}

    /* begin ABackpropLayer interface */
    virtual std::string tag() const { return "FullyConnectedLayer"; }

    /// Initialize synaptic weights.
    virtual void init(std::unique_ptr<RNG> &rng, WeightInit init) {
        init(rng, *weights, nInputs, nOutputs);
        init(rng, *bias, nInputs, nOutputs);
    }

    virtual std::shared_ptr<Array> getInputBuffer() {
        return shared.inputBuffer;
    }

    virtual std::shared_ptr<Array> getOutputBuffer() {
        return shared.outputBuffer;
    }

    virtual std::shared_ptr<const Array> getWeights() const {
        return weights;
    }
    virtual std::shared_ptr<const Array> getBias() const {
        return bias;
    }
    virtual std::shared_ptr<const Array> getWeightSensitivity() const {
        return weightSensitivity;
    }
    virtual std::shared_ptr<const Array> getBiasSensitivity() const {
        return biasSensitivity;
    }
    virtual const Activation &getActivation() const {
        return *activationFunction;
    }

    virtual std::shared_ptr<ABackpropLayer> clone() const {
        return makeClone();
    }

    virtual size_t inputDim() const {
        return nInputs;
    }
    virtual size_t outputDim() const {
        return nOutputs;
    }

    virtual const Array &output(const Array &inputs) {
        shared.allocate(nInputs, nOutputs); // just in case user didn't init()
        outputInplace(inputs);
        return *shared.outputBuffer;
    }

    virtual const Array &backprop(const Array &errors) {
        shared.allocate(nInputs, nOutputs); // just in case user didn't init()
        backpropInplace(errors);
        return propagatedErrors;
    }

    virtual void setLearningPolicy(const ALearningPolicy &lp) {
        policy = lp.clone();
    }

    virtual void update() {
        if (!policy) {
            throw std::logic_error("learning policy is not defined");
        }
        policy->update(*weights, *bias, *weightSensitivity, *biasSensitivity);
    }
    /* end ABackpropLayer interface */
};


/** Apply Activation to all inputs.
 *
 * This layer doesn't have any parameters (weights). */
class ActivationLayer : public ABackpropLayer {
protected:
    size_t nSize; // the number of input and outputs is the same
    const Activation *activationFunction; //< $\phi$
    Array activationGrad; //< $\phi^\prime(v)$
    Array propagatedErrors;

    std::shared_ptr<ActivationLayer> makeClone() const {
        auto p = std::make_shared<ActivationLayer>(*this);
        if (!p) {
            throw std::runtime_error("cannot clone ActivationLayer");
        }
        p->shared = shared.clone();
        return p;
    }

    virtual void outputInplace(const Array &inputs) {
        Array &outputs = *shared.outputBuffer;
        if (outputs.size() != nSize) {
            outputs.resize(nSize);
        }
        *shared.inputBuffer = inputs;
        auto &activation = (*activationFunction);
        std::transform(
            std::begin(inputs), std::end(inputs),
            std::begin(outputs),
            [&](float y) { return activation(y); });
    }

    virtual void backpropInplace(const Array &errors) {
        auto &inputs = *shared.inputBuffer;
        auto &activation = (*activationFunction);
        std::transform(
            std::begin(inputs), std::end(inputs),
            std::begin(activationGrad),
            [&](float y) { return activation.derivative(y); });
        propagatedErrors = activationGrad * errors;
    }

public:
    ActivationLayer(size_t n, const Activation &af)
        : nSize(n), activationFunction(&af),
          activationGrad(n), propagatedErrors(n) {}

    /* begin ABackpropLayer interface */
    virtual std::string tag() const { return "ActivationLayer"; }

    // this layer doesn't have parameters, nothing to initialize
    virtual void init(std::unique_ptr<RNG> &, WeightInit) {}
    virtual std::shared_ptr<Array> getInputBuffer() { return shared.inputBuffer; }
    virtual std::shared_ptr<Array> getOutputBuffer() { return shared.outputBuffer; }
    virtual std::shared_ptr<ABackpropLayer> clone() const { return makeClone(); }
    virtual size_t inputDim() const { return nSize; }
    virtual size_t outputDim() const { return nSize; }
    virtual const Array &output(const Array &inputs) {
        shared.allocate(nSize, nSize); // just in case user didn't init()
        assert (shared.ready());
        assert (nSize == inputs.size());
        outputInplace(inputs);
        return *shared.outputBuffer;
    }
    virtual const Array &backprop(const Array &errors) {
        shared.allocate(nSize, nSize); // just in case user didn't init()
        assert (shared.ready());
        assert (nSize == errors.size());
        backpropInplace(errors);
        return propagatedErrors;
    }
    virtual void setLearningPolicy(const ALearningPolicy &) {} // do nothing
    virtual void update() {} // do nothing
    /* end of ABackpropLayer interface */
};


/* Loss Layers
 * -----------
 */

/** Euclidean loss.
 *
 * Loss is the Euclidean distance between two vectors:
 *
 * $$ E_2(\mathbf{y}, \mathbf{d}) = \sqrt{ \sum_i (y_i - d_i)^2 } $$
 *
 * This loss layer may be used for regressing real-valued labels.
 * Minimizing Euclidean loss $E(y,d)$ means predicting
 * the conditional mean of $d$.
 *
 * References:
 *
 *  - https://en.wikipedia.org/wiki/Convolutional_neural_network#Loss_layer
 *  - Rosasco, et al. (2004) Are loss functions all the same? In: Neural Computation
 *  - Langford (2007) Loss Function Semantics. Online: http://hunch.net/?p=269
 */
class EuclideanLoss : public ALossLayer {
protected:
    size_t nSize;
    Array lossGrad; //< $\partial E/\partial y_i$

public:
    EuclideanLoss(size_t n) : nSize(n), lossGrad(0.0, n) {}

    virtual std::string tag() const { return "EuclideanLoss"; }

    virtual float output(const Array &actual, const Array &expected) {
        float lossSquared = 0.0;
        assert (nSize == actual.size());
        assert (nSize == expected.size());
        for (size_t i = 0; i < nSize; ++i) {
            float delta = expected[i] - actual[i];
            lossSquared += (delta*delta);
            lossGrad[i] = delta;
        }
        return std::sqrt(lossSquared);
    }

    virtual const Array &backprop() {
        return lossGrad;
    }

    virtual std::shared_ptr<ALossLayer> clone() const {
        auto c = std::make_shared<EuclideanLoss>(*this);
        c->shared = shared.clone();
        return c;
    }

    virtual size_t inputDim() const { return nSize; }
};


/** Softmax activation with multinomial cross-entropy loss.
 *
 * Softmax function is a generalization of the logistic function and
 * produces outputs in the range (0,1) which sum to 1.
 * Given inputs $y_i$, the softmax function $\phi_j$ is defined as
 *
 * $$ \phi(\mathbf{y})_j = \frac {\exp{y_j}} {\sum_i \exp{y_i}} $$
 *
 * It is often used as the last layer in the multiclass classification
 * problems using a cross-entropy loss function. Class labels should be
 * non-negative and sum to 1.
 *
 * For a multiclass classification problem with $K$ different classes
 * where $d_i$ is the true (desired) propability that the sample belongs
 * to class $i$, and $\phi_i$ is the predicted probability, the
 * cross-entropy loss function is defined as
 *
 * $$ E(\mathbf{\phi}, \mathbf{d}) = - \sum_{i = 1}^{K} d_i \ln \phi_i $$
 *
 * OneHotEncoder may be used to encode categorical labels for use with
 * the cross-entropy loss function.
 *
 * References:
 *
 *  - PRML, 4.3.4 Multiclass logistic regression, Page 209.
 *  - http://stats.stackexchange.com/q/79454
 *  - http://en.wikipedia.org/wiki/Softmax_function
 *  - https://en.wikipedia.org/wiki/Cross_entropy
 */
class SoftmaxWithLoss : public ALossLayer {
protected:
    size_t nSize;
    Array softmaxOutput; //< $\phi(\mathbf{y})_i$
    Array lossGrad; //< $\partial E/\partial y_i$

    /** Softmax activation. */
    static Array softmax(const Array &input) {
        Array softmaxOutput(0.0, input.size());
        size_t nSize = input.size();
        // a trick to avoid unbounded exponents
        float maxInput = std::numeric_limits<float>::min();
        for (size_t i = 0; i < nSize; ++i) {
            if (input[i] > maxInput) {
                maxInput = input[i];
            }
        }
        // calculate exponents
        float expTotal = 0.0;
        for (size_t i = 0; i < nSize; ++i) { // calculate exponents
            softmaxOutput[i] = std::exp(input[i] - maxInput);
            expTotal += softmaxOutput[i];
        }
        // normalize exponents
        float expTotalInv = 1.0/expTotal;
        softmaxOutput *= expTotalInv;
        return softmaxOutput;
    }

public:
    SoftmaxWithLoss(size_t n) : nSize(n), softmaxOutput(n), lossGrad(n) {}

    virtual std::string tag() const { return "SoftmaxWithLoss"; }

    virtual float output(const Array &actual, const Array &expected) {
        float lossTotal = 0.0;
        assert (nSize == actual.size());
        assert (nSize == expected.size());
        softmaxOutput = softmax(actual);
        for (size_t i = 0; i < nSize; ++i) {
            // accumulate loss across all class labels
            lossTotal -= expected[i] * std::log(softmaxOutput[i]);
            // combination of softmax activation with cross-entropy loss
            // allows to simplify loss gradient calculation:
            //
            // $$ \grad_y E = d - \phi(y)$$
            lossGrad[i] = expected[i] - softmaxOutput[i];
        }
        return lossTotal;
    }

    virtual const Array &backprop() {
        return lossGrad;
    }

    virtual std::shared_ptr<ALossLayer> clone() const {
        auto c = std::make_shared<SoftmaxWithLoss>(*this);
        c->shared = shared.clone();
        return c;
    }

    virtual size_t inputDim() const { return nSize; }
};


/** Hinge loss for binary classification.
 *
 * For a binary classification problem with true (desired) labels
 * $d = +1$ and $d = -1$, and classifier output $y$, the hinge
 * loss is defined as
 *
 * $$ E(y, d) = \max(0, 1 - d y) $$
 */
class HingeLoss : public ALossLayer {
protected:
    Array lossGrad;

public:
    HingeLoss() : lossGrad(1) {}

    virtual std::string tag() const { return "HingeLoss"; }

    virtual float output(const Array &actual, const Array &expected) {
        assert (1 == actual.size());
        assert (1 == expected.size());
        float loss;
        float p = actual[0]*expected[0];
        if (p >= 1.0) {
            lossGrad[0] = 0.0;
            loss = 0.0;
        } else {
            lossGrad[0] = - expected[0];
            loss = 1 - p;
        }
        return loss;
    }

    virtual const Array &backprop() {
        return lossGrad;
    }

    virtual std::shared_ptr<ALossLayer> clone() const {
        auto c = std::make_shared<HingeLoss>(*this);
        c->shared = shared.clone();
        return c;
    }

    virtual size_t inputDim() const { return 1; }
};


/** A feed-forward neural network.
 *
 * It is recommended to use MakeNet builder class to
 * construct and configure Nets. */
class Net {
protected:
    std::vector<std::shared_ptr<ABackpropLayer>> layers;
    std::shared_ptr<ALossLayer> lossLayer = nullptr;

    void selfCheck() const {
        if (layers.empty()) {
            throw std::logic_error("no layers");
        }
        if (!lossLayer) {
            throw std::logic_error("Net has no loss layer");
        }
    }

public:
    Net() : layers(0) {}

    virtual Net &append(std::shared_ptr<ABackpropLayer> layer) {
        layers.push_back(std::move(layer));
        if (layers.size() >= 2) { // connect the last two layers
           auto n = layers.size();
           connect(*layers[n-2], *layers[n-1]);
        }
        return *this;
    }

    virtual Net &append(const ABackpropLayer &layer) {
        return append(layer.clone());
    }

    virtual Net &append(std::shared_ptr<ALossLayer> loss) {
        if (layers.empty()) {
            throw std::logic_error("cannot append loss layer to an empty Net");
        }
        if (!this->lossLayer) {
            // TODO: check loss layer shape and connect the last layer
            // TODO: prevent appending
            this->lossLayer = loss;
            connect(*layers.back(), *lossLayer);
        } else {
            throw std::logic_error("cannot append another loss layer");
        }
        return *this;
    }

    virtual Net &append(const ALossLayer &loss) {
        return append(loss.clone());
    }

    virtual void
    init(std::unique_ptr<RNG> &rng, WeightInit init = Init::normalXavier) {
        for (size_t i = 0u; i < layers.size(); ++i) {
            layers[i]->init(rng, init);
        }
    }

    virtual void clear() { layers.clear(); }

    const Array &output(const Array &inputs) {
        selfCheck();
        const Array *out = &(layers[0]->output(inputs));
        for (auto i = 1u; i < layers.size(); ++i) {
            out = &(layers[i]->output(*out));
        }
        return *out;
    }

    // TODO: checks on order of calls + tests (loss and backprop after output)
    // TODO: optimize: remember last output and don't recalculate it
    float loss(const Array &inputs, const Array &expected) {
        auto &out = output(inputs);
        return lossLayer->output(out, expected);
    }

    std::tuple<const Array &, float>
    outputWithLoss(const Array &inputs, const Array &expected) {
        const Array &out = output(inputs);
        float lossValue = lossLayer->output(out, expected);
        return std::make_tuple(out, lossValue);
    }

    const Array &backprop(const Array &errors) {
        if (layers.empty()) {
            throw std::logic_error("no layers");
        }
        const Array *bpr = &errors;
        size_t n = layers.size();
        for (size_t back_offset = 0; back_offset < n; ++back_offset) {
            size_t i = n - 1 - back_offset;
            bpr = &(layers[i]->backprop(*bpr));
        }
        return *bpr;
    }

    const Array &backprop() {
        selfCheck();
        return backprop(lossLayer->backprop());
    }

    void setLearningPolicy(const ALearningPolicy &lp) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->setLearningPolicy(lp);
        }
     }

    void update() {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update();
        }
    }

    size_t size() const { return layers.size(); }
    std::shared_ptr<const ABackpropLayer> getLayer(size_t i) const {
       return (i < layers.size()) ? layers[i] : nullptr;
    }
    std::shared_ptr<ABackpropLayer> getLayer(size_t i) {
       return (i < layers.size()) ? layers[i] : nullptr;
    }
    std::shared_ptr<const ALossLayer> getLossLayer() const { return lossLayer; }
    std::shared_ptr<ALossLayer> getLossLayer() { return lossLayer; }
};


/// Layer factory.
class MakeLayer {
public:
    size_t nInputs = 0;
    size_t nOutputs = 0;
    std::shared_ptr<Array> maybeWeights = nullptr;
    std::shared_ptr<Array> maybeBias = nullptr;
    const Activation *maybeActivation = nullptr;

    MakeLayer() {}
    MakeLayer(size_t n)
        : nInputs(n), nOutputs(n) {}
    MakeLayer(size_t n, const Activation &af)
        : nInputs(n), nOutputs(n), maybeActivation(&af) {}
    MakeLayer(size_t nInputs, size_t nOutputs)
        : nInputs(nInputs), nOutputs(nOutputs) {}
    MakeLayer(size_t nInputs, size_t nOutputs, const Activation &af)
        : nInputs(nInputs), nOutputs(nOutputs), maybeActivation(&af) {}
    MakeLayer(const Array &weights, const Array &bias)
        : maybeWeights(std::shared_ptr<Array>(new Array(weights))),
          maybeBias(std::shared_ptr<Array>(new Array(bias))) {}
    MakeLayer(const Array &weights, const Array &bias, const Activation &af)
        : maybeWeights(std::shared_ptr<Array>(new Array(weights))),
          maybeBias(std::shared_ptr<Array>(new Array(bias))),
          maybeActivation(&af) {}

    MakeLayer &setInputDim(size_t n) {
        nInputs = n;
        return *this;
    }
    MakeLayer &setOutputDim(size_t n) {
        nOutputs = n;
        return *this;
    }
    MakeLayer &setWeights(Array &w) {
        maybeWeights = std::make_shared<Array>(w);
        return *this;
    }
    MakeLayer &setBias(Array &b) {
        maybeBias = std::make_shared<Array>(b);
        return *this;
    }
    MakeLayer &setActivation(const Activation &af) {
        maybeActivation = &af;
        return *this;
    }

    /// create a new FullyConnectedLayer
    std::shared_ptr<FullyConnectedLayer> FC() {
        if (!maybeActivation) {
            maybeActivation = &linearActivation;
        }
        if (maybeWeights && maybeBias) {
            return std::make_shared<FullyConnectedLayer>
                (*maybeWeights, *maybeBias, *maybeActivation);
        } else if (nInputs && nOutputs) {
            return std::make_shared<FullyConnectedLayer>
                (nInputs, nOutputs, *maybeActivation);
        } else {
            logic_error("a FullyConnectedLayer");
            return nullptr; // unreachable
        }
    }

    /// create a new ActivationLayer
    std::shared_ptr<ActivationLayer> activation() {
        if (!maybeActivation) {
            maybeActivation = &linearActivation;
        }
        if (nOutputs && !nInputs) {
            nInputs = nOutputs;
        }
        if (nInputs) {
            return std::make_shared<ActivationLayer>(nInputs, *maybeActivation);
        } else {
            logic_error("an ActivationLayer");
            return nullptr; // unreachable
        }
    }

    /// create a new EuclideanLoss layer
    std::shared_ptr<EuclideanLoss> L2Loss() {
        if (nInputs) {
            return std::make_shared<EuclideanLoss>(nInputs);
        } else {
            logic_error("an EuclideanLoss");
            return nullptr; // unreachable
        }
     }

    /// create a new SoftmaxWithLoss layer
    std::shared_ptr<SoftmaxWithLoss> softmax() {
        if (nInputs) {
            return std::make_shared<SoftmaxWithLoss>(nInputs);
        } else {
            logic_error("a SoftmaxWithLoss layer");
            return nullptr; // unreachable
       }
    }

    /// create a new HingeLoss layer
    std::shared_ptr<HingeLoss> hinge() {
        if (nInputs) {
            return std::make_shared<HingeLoss>();
        } else {
            logic_error("a HingeLoss");
            return nullptr; // unreachable
        }
     }

private:
    void logic_error(std::string what) {
        auto m = "cannot create " + what + ": invalid configuration";
        throw std::logic_error(m);
     }
};


class MakeNet {
protected:
    size_t nInputs;
    size_t nOutputs;

    enum class LayerType { FC, Activation, L2Loss, Softmax, Hinge };
    std::vector<MakeLayer> layerMakers;
    std::vector<LayerType> layerTypes;
    bool hasLoss = false;

public:
    /// Create a new net.
    MakeNet(size_t nInputs = 0)
        : nInputs(nInputs), nOutputs(nInputs), layerMakers(0), layerTypes(0) {}

    /// Set the number of nodes in the input layer.
    MakeNet &setInputDim(size_t n) {
        if (!layerMakers.empty()) {
            throw std::logic_error("cannot setInputDim() with layers above");
        }
        nInputs = n;
        nOutputs = n;
        return *this;
    }

    /// Append a FullyConnectedLayer.
    MakeNet &addFC(size_t n, const Activation &af = scaledTanh) {
        checkConfig();
        if (!n) {
            throw std::logic_error("cannot add a layer with zero outputs");
        }
        MakeLayer mk(nOutputs, n, af);
        layerMakers.push_back(mk);
        layerTypes.push_back(LayerType::FC);
        nOutputs = n;
        return *this;
    }

    /// Append an initialized FullyConnectedLayer.
    MakeNet &addFC(const Array &weights, const Array &bias,
                   const Activation &af = scaledTanh) {
        size_t w_rows = bias.size();
        size_t w_cols = weights.size() / w_rows;
        if (w_cols != nOutputs) {
            throw std::logic_error("cannot add a layer with an incompatible weight matrix");
        }
        checkConfig();
        MakeLayer mk(weights, bias, af);
        layerMakers.push_back(mk);
        layerTypes.push_back(LayerType::FC);
        nOutputs = w_rows;
        return *this;
    }

    /// Append an ActivationLayer.
    MakeNet &addActivation(const Activation &af) {
        checkConfig();
        MakeLayer mk(nOutputs, af);
        layerMakers.push_back(mk);
        layerTypes.push_back(LayerType::FC);
        return *this;
    }

    /// Append an EuclideanLoss layer.
    MakeNet &addL2Loss() {
        return addLoss(LayerType::L2Loss);
    }

    /// Append a SoftmaxWithLoss layer (softmax activation + cross-entropy loss).
    MakeNet &addSoftmax() {
        return addLoss(LayerType::Softmax);
    }

    /// Append a HingeLoss layer.
    MakeNet &addHingeLoss() {
        return addLoss(LayerType::Hinge);
    }

    /// Configure a new multilayer perceptron (a stack of 'FullyConnectedLayer's).
    ///
    /// @param shape   {number_of_input_nodes, layer_1_size, ..., layer_N_size}
    /// @param af      activation function
    MakeNet &MultilayerPerceptron(std::vector<size_t> shape,
                                  const Activation &af = scaledTanh) {
        if (shape.size() < 2) {
            throw std::invalid_argument("shape should define at least one layer");
        }
        if (layerMakers.empty()) {
            setInputDim(shape[0]);
        } else {
            if (nOutputs != shape[0]) {
                throw std::logic_error("cannot add an MLP with wrong inputDim");
            }
        }
        for (size_t i = 1; i < shape.size(); ++i) {
            addFC(shape[i], af);
        }
        return *this;
    }

    /// Create a new feed-forward neural net.
    Net make() {
        Net net;
        for (size_t i = 0; i < layerMakers.size(); ++i) {
            auto ltype = layerTypes[i];
            auto lmaker = layerMakers[i];
            switch (ltype) {
                case LayerType::FC:
                    net.append(lmaker.FC()); break;
                case LayerType::Activation:
                    net.append(lmaker.activation()); break;
                case LayerType::L2Loss:
                    net.append(lmaker.L2Loss()); break;
                case LayerType::Softmax:
                    net.append(lmaker.softmax()); break;
                case LayerType::Hinge:
                    net.append(lmaker.hinge()); break;
                default:
                    throw std::logic_error("unsupported layer type"); break;
            }
        }
        return net;
    }

    /// Create and initialize a new feed-forward neural net.
    Net init(std::unique_ptr<RNG> &rng, WeightInit init = Init::normalXavier) {
        Net net = make();
        net.init(rng, init);
        return net;
    }

    /// Create and initialize a new feed-forward neural net.
    ///
    /// This version of init() method creates, seeds, uses, and then discards a
    /// temporary random number generator.
    Net init(WeightInit init = Init::normalXavier) {
        std::unique_ptr<RNG> rng(Init::newRNG());
        Net net = this->init(rng, init);
        return net;
    }

private:
    void checkConfig() {
        if (!nInputs) {
            throw std::logic_error("cannot add a layer before setInputDim()");
        }
        if (hasLoss) {
            throw std::logic_error("cannot add a layer after a loss layer");
        }
    }

    MakeNet &addLoss(LayerType lt) {
        checkConfig();
        MakeLayer mk(nOutputs);
        layerMakers.push_back(mk);
        layerTypes.push_back(lt);
        nOutputs = 0;
        hasLoss = true;
        return *this;
    }
};


// TODO: CNN layer
// TODO: max-pooling layer
// TODO: NN builder which takes Ciresan's string-like specs: 100c5-mp2-...
// TODO: sliding window search for CNNs


/** EpochCallback structure wraps a function to be called every N-th epoch.
 *
 * If period is zero, then callback is disabled.
 * If period is one, then the callback is called every epoch.
 * To invoke the callback use operator().
 * callback function may return true to stop training early. */
struct EpochCallback {
    int period;
    std::function<bool(int epoch)> callback; //< return true to stop early
    bool operator()(int currentEpoch, bool alwaysInvoke = false) const {
        if (alwaysInvoke || (period > 0 && currentEpoch % period == 0)) {
            return callback(currentEpoch);
        } else {
            return false;
        }
    }
};

/** IterationCallback structure wraps a function to be called every N-th iteration.
 *
 * If period is zero, then callback is disabled.
 * If period is one, then the callback is called every iteration.
 * To invoke the callback use operator().
 * callback function may return true to stop training early. */
struct IterationCallback {
    int period;
    std::function<bool(int iteration)> callback; //< return true to stop early
    bool operator()(int currentIteration, bool alwaysInvoke = false) const {
        if (alwaysInvoke || (period > 0 && currentIteration % period == 0)) {
            return callback(currentIteration);
        } else {
            return false;
        }
    }
};

const EpochCallback noEpochCallback {0, [](int) { return false; }};

const IterationCallback noIterationCallback {0, [](int) { return false; }};


/** Stochastic gradient descent. */
class SGD {
public:
    /** Train using stochastic gradient descent.
     *
     * Use net.setLearningPolicy() to change learning parameters of the network
     * _before_ calling `SGD::train`.
     *
     * @param rng             Random number generator for shuffling.
     * @param net             Neural network to be trained.
     * @param trainSet        Training set.
     * @param epochs          How many iterations (epochs) to run;
     *                        the entire training set is processed once per epoch.
     * @param epochCallback   A callback to invoke before every N-th epoch.
     *                        The callback is also called after the last iteration.
     *                        epochCallback.callback may return true to stop early.
     * @param sampleCallback  A callback to invoke after every N-th sample.
     *                        sampleCallback.callback may return true to stop early.
     *
     * See:
     *
     *  - Efficient BackProp (2012) LeCun et al. In: NNTT.
     *    http://cseweb.ucsd.edu/classes/wi08/cse253/Handouts/lecun-98b.pdf
     */
    static
    void train(std::unique_ptr<RNG> &rng,
            Net &net, LabeledDataset &trainSet, int epochs,
            const EpochCallback &epochCallback = noEpochCallback,
            const IterationCallback &sampleCallback = noIterationCallback) {
        for (int j = 0; j < epochs; ++j) {
            if (epochCallback(j)) {
                return;
            }
            trainSet.shuffle(rng);
            int sampleCounter = 0;
            for (auto sample : trainSet) {
                net.loss(sample.data, sample.label);
                net.backprop();
                net.update();
                if (sampleCallback(sampleCounter)) {
                    return;
                }
                ++sampleCounter;
            }
        }
        epochCallback(epochs, true);
    }

    /** Train using stochastic gradient descent.
     *
     * This method creates, seeds, uses and then discards
     * a temporary random number generator. Otherwise this function is
     * identical to SGD::train which takes also an 'rng' parameter. */
    static
    void train(
            Net &net, LabeledDataset &trainSet, int epochs,
            const EpochCallback &epochCallback = noEpochCallback,
            const IterationCallback &sampleCallback = noIterationCallback) {
        std::unique_ptr<RNG> rng(Init::newRNG());
        SGD::train(rng, net, trainSet, epochs, epochCallback, sampleCallback);
    }
};

} // end of namespace notch

#endif /* NOTCH_H */
