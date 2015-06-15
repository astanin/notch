#ifndef NOTCH_PRE_H
#define NOTCH_PRE_H

/// @file notch_pre.hpp Optional data preprocessing

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

#include <cmath>       // round
#include <functional>
#include <map>
#include <set>
#include <valarray>
#include <vector>
#include <sstream>

#include "notch.hpp"

// TODO: StandardScaler (mu, sigmae)
// TODO: MinMaxScaler
// TODO: Normalizer
// TODO: decorrelation for elements
// TODO: Dataset shuffle
// TODO: train and test set splitter
// TODO: K-fold split

#ifdef ENABLE_GCC49_CYGWIN_MSYS2_WORKAROUND
// std::round is not available in some variants of GNU C++ 4.9 on Windows
// http://stackoverflow.com/q/28997206/25450
#include <math.h>
namespace std {
    float round(float x) { return round(x); }
}
#endif /* ENABLE_GCC49_CYGWIN_MSYS2_WORKAROUND */

/** Encode categorical features as vectors.
 *
 * N distinct categorical features are mapped to N-dimensional vectors,
 * where all elements are zero except one.  For instance, [1,2,3,1] are can be
 * mapped to [[1,0,0],[0,1,0],[0,0,1],[1,0,0]].  This encoding may be
 * particularly useful in multi-class classification.
 **/
class OneHotEncoder : public ADatasetTransformer {
    private:
    std::vector<int> inputColumns; // categorical columns to be transformed
    std::map<int, std::vector<int>> columnValues; // columnIndex -> [categories]
    size_t nInputCols = 0; // how many columns there're in original data
    size_t nExtraCols = 0; // how many columns will be added

    void countColumnValues(const Dataset &d) {
        std::map<int, std::set<int>> colMaps; // columnIndex -> {categories}
        for (Array a : d) {
            for (int colId : inputColumns) {
                colMaps[colId].insert(int(std::round(a[colId])));
            }
        }
        nExtraCols = 0;
        columnValues.clear();
        for (auto col : colMaps) {
            auto colId = col.first;
            auto colVals = col.second;
            columnValues[colId] = std::vector<int>(colVals.begin(), colVals.end());
            nExtraCols += columnValues[colId].size() - 1;
        }
    }

public:

    /** Create a OneHot encoder.
     *
     * The encoder has to be configured using `OneHotEncoder::fit()` method.
     *
     * @param columns   a list of categorical columns to encode;
     *                  if empty, encode all columns;
     *                  negative indices are allowed (-1 is the last column)
     **/
    OneHotEncoder(std::vector<int> columns = {})
        : inputColumns(columns) {}

    /** Create and configure a OneHot encoder using a `Dataset`.
     *
     * @param data      a `Dataset` which is used to learn possible columns' values
     * @param columns   a list of categorical columns to encode;
     *                  if empty, encode all columns;
     *                  negative indices are allowed (-1 is the last column)
     **/
    OneHotEncoder(const Dataset &data, std::vector<int> columns = {})
      : inputColumns(columns) {
        fit(data);
    }

    ADatasetTransformer &fit(const Dataset &d) {
        if (d.empty()) {
            return *this;
        }
        nInputCols = d[0].size();
        if (inputColumns.empty()) { // transform all columns
            for (size_t i = 0; i < nInputCols; ++i) {
                inputColumns.push_back(i);
            }
        } else { // change negative column indices, if any, to positive
            for (size_t i = 0; i < inputColumns.size(); ++i) {
                int colId = inputColumns[i];
                inputColumns[i] = ((colId + nInputCols) % nInputCols);
            }
        }
        countColumnValues(d);
        return *this;
    }

    virtual Dataset apply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(apply(a));
        }
        return dOut;
    }

    virtual Array apply(const Array &input) {
        if (input.size() != nInputCols) {
            std::ostringstream ss;
            ss << "input.size(): " << input.size()
               << ", expected: " << nInputCols;
            throw std::invalid_argument(ss.str());
        }
        Array output(0.0, input.size() + nExtraCols);
        size_t outCol = 0;
        for (size_t inCol = 0; inCol < input.size(); ++inCol) {
            if (columnValues.count(inCol)) { // replace many columns
                int inValue = int(std::round(input[inCol]));
                for (auto v : columnValues[inCol]) {
                    output[outCol] = int(v == inValue);
                    ++outCol;
                }
            } else { // keep this column as is
                output[outCol] = input[inCol];
                ++outCol;
            }
        }
        return output;
    }

    virtual Dataset unapply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(unapply(a));
        }
        return dOut;
    }

    virtual Array unapply(const Array &input) {
        if (input.size() != nInputCols + nExtraCols) {
            std::ostringstream ss;
            ss << "input.size(): " << input.size()
               << ", expected: " << nInputCols + nExtraCols;
            throw std::invalid_argument(ss.str());
        }
        Array output(0.0, input.size() - nExtraCols);
        size_t inCol = 0;
        for (size_t outCol = 0; outCol < output.size(); ++outCol) {
            if (columnValues.count(outCol)) { // replace with one column
                // find the column with the largest output
                size_t nColsPerFeature = columnValues[outCol].size();
                float maxVal = std::numeric_limits<float>::lowest();
                size_t maxIdx = 0;
                for (size_t i = 0; i < nColsPerFeature; ++i) {
                    if (input[inCol + i] > maxVal) {
                        maxVal = input[inCol + i];
                        maxIdx = i;
                    }
                }
                // use the category corresponding to the largest value
                output[outCol] = columnValues[outCol][maxIdx];
                inCol += nColsPerFeature;
            } else {
                output[outCol] = input[inCol];
                ++inCol;
            }
        }
        return output;
    }
};

/** SquareAugmented transformer is a way to implement a SQUARE-MLP (SMLP) network.
 *
 * For each input sample with $n$ components @f$ {x_1, ..., x_n} @f$,
 * the transformer creates a new sample with @f$ 2 n @f$ components
 * @f$ {x_1, ..., x_n, x_1^2, ... x_n^2} @f$, with the extra @f$ n @f$ inputs
 * set equal to the squared of the original inputs.
 *
 * This transform has the same effect as implementing a "SQuare Unit Augmented,
 * Radially Extended, Multilayer Perceptron (SQUARE-MLP or SMLP)".
 * This architecture can efficiently capture not only global dataset
 * features like normal MLPs, but also local features similar to radial basis
 * function networks (RBFNs).
 *
 * This technique may be used if the target function to be approximated
 * has localized "bumps" or narrow ridges.
 *
 * See also Two-Spiral example.
 *
 * References:
 *
 *  - Flake, G. W. (2012) Square Unit Augmented, Radially Extended,
 *    Multilayer Perceptron. In: NNTT.
 */
class SquareAugmented : public ADatasetTransformer {
public:
    virtual Dataset apply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(apply(a));
        }
        return dOut;
    }

    virtual Array apply(const Array &input) {
        size_t n = input.size();
        Array output(0.0, 2 * n);
        auto in = std::begin(input);
        auto inEnd = std::end(input);
        auto out = std::begin(output);
        auto outSquare = out + n;
        for (; in != inEnd; ++in, ++out, ++outSquare) {
            float x = *in;
            *out = x;
            *outSquare = x * x;
        }
        return output;
    }

    virtual Dataset unapply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(unapply(a));
        }
        return dOut;
    }

    virtual Array unapply(const Array &input) {
        size_t n = input.size();
        Array output(0.0, n / 2);
        std::copy(std::begin(input),
                  std::begin(input) + (n/2),
                  std::begin(output));
        return output;
    }
};


class CustomTransform : public ADatasetTransformer {
protected:
    std::function<Array(const Array &)> transform;
    std::function<Array(const Array &)> inverseTransform;

public:
    CustomTransform(std::function<Array(const Array &)> transform,
                    std::function<Array(const Array &)> inverseTransform)
        : transform(transform), inverseTransform(inverseTransform) {}

    virtual Dataset apply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(apply(a));
        }
        return dOut;
    }

    virtual Array apply(const Array &input) {
        Array output = transform(input);
        return output;
    }

    virtual Dataset unapply(const Dataset &dataIn) {
        Dataset dOut;
        dOut.reserve(dataIn.size());
        if (dataIn.empty()) {
            return dataIn;
        }
        for (Array a : dataIn) {
            dOut.push_back(unapply(a));
        }
        return dOut;
    }

    virtual Array unapply(const Array &input) {
        Array output = inverseTransform(input);
        return output;
    }
};


#endif /* NOTCH_PRE_H */
