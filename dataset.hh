#ifndef DATASET_H
#define DATASET_H


#include <sstream>
#include <ostream>
#include <vector>
#include <valarray>
#include <iterator>   // begin, end
#include <functional> // function<>
#include <assert.h>
#include <initializer_list>
#include <typeinfo>   // typeid


/**
 * Data types
 * ==========
 *
 * Neural network consumes a vector of numerical values, and produce a vector
 * of numerical outputs. Without the loss of generality we may consider them
 * arrays of double precision floating point numbers. 53-bit integers can be
 * represented by double too, it should be enough for many applications.
 *
 * We use C++ `valarray` type to make code more concise and expressive
 * (valarrays implement elementwise operations).
 */

using Input = std::valarray<double>;
using Output = std::valarray<double>;


/** Supervised learning requires labeled data.
 *  Labels are vector of numeric values. */
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

    /// load a dataset from file (the same format as FANN)
    void readFANN(std::istream &in) {
        in >> nSamples >> inputDimension >> outputDimension;
        inputs.clear();
        outputs.clear();
        for (size_t i = 0; i < nSamples; ++i) {
            Input input(inputDimension);
            Output output(outputDimension);
            for (size_t j = 0; j < inputDimension; ++j) {
                in >> input[j];
            }
            for (size_t j = 0; j < outputDimension; ++j) {
                in >> output[j];
            }
            inputs.push_back(input);
            outputs.push_back(output);
        }
    }

public:
    /// An iterator to process all labeled data samples.
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

    LabeledDataset() : nSamples(0), inputDimension(0), outputDimension(0) {}
    LabeledDataset(std::istream &in) { readFANN(in); }
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

    friend std::istream &operator>>(std::istream &out, LabeledDataset &ls);
    friend std::ostream &operator<<(std::ostream &out, const LabeledDataset &ls);
};

/** Input-output
 *  ------------
 *
 *  Input and output values are space-separated lines.*/
std::ostream &operator<<(std::ostream &out, const Input &xs) {
    for (auto it = std::begin(xs); it != std::end(xs); ++it) {
        if (it != std::begin(xs)) {
            out << " ";
        }
        out << *it;
    }
    return out;
}

/** Labeled pairs are split in two lines. */
std::ostream &operator<<(std::ostream &out, const LabeledData &p) {
    out << p.data << "\n" << p.label;
    return out;
}


/** `LabeledDataset`'s input format is compatible with FANN library. */
std::istream &operator>>(std::istream &in, LabeledDataset &ls) {
    ls.readFANN(in);
    return in;
}

/** `LabeledDataset`'s output format is compatible with FANN library. */
std::ostream &operator<<(std::ostream &out, const LabeledDataset &ls) {
    out << ls.nSamples << " "
        << ls.inputDimension << " "
        << ls.outputDimension << "\n";
    for (auto sample : ls) {
        out << sample << "\n";
    }
    return out;
}

#endif /* DATASET_H */

