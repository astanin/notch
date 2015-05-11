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


/** In practice, we often store data and labels separately.
 * In this case we may want to construct `LabeledData` values on the fly,
 * when iterating two sequences together. */
class LabeledDatasIterator : public std::iterator<std::input_iterator_tag, LabeledData> {
private:
    std::vector<Input> const *inputs;
    std::vector<Output> const *outputs;
    size_t position;

public:
    // TODO: take iterators, like std::transform
    LabeledDatasIterator(const std::vector<Input> &inputs,
                         const std::vector<Output> &outputs,
                         size_t position = 0)
        : inputs(&inputs), outputs(&outputs), position(position){};

    bool operator==(const LabeledDatasIterator &rhs) const {
        return (inputs == rhs.inputs &&
                outputs == rhs.outputs &&
                position == rhs.position);
    }

    bool operator!=(const LabeledDatasIterator &rhs) const {
        return !(*this == rhs);
    }

    LabeledData operator*() const {
        LabeledData lp{inputs->at(position), outputs->at(position)};
        return lp;
    }

    LabeledDatasIterator &operator++() {
        if (position < inputs->size()) {
            ++position;
        }
        return *this;
    }

    LabeledDatasIterator &operator++(int) { return ++(*this); }
};


/** A `LabeledSet` consists of multiple `LabeledData` samples.
 *  `LabeledSet`s can be used like training or testing sets.
 */
class LabeledSet {
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
    LabeledSet() : nSamples(0), inputDimension(0), outputDimension(0) {}
    LabeledSet(std::istream &in) { readFANN(in); }
    LabeledSet(std::initializer_list<LabeledData> samples)
        : nSamples(0), inputDimension(0), outputDimension(0) {
        for (LabeledData s : samples) {
            append(s);
        }
    }

    LabeledDatasIterator begin() const {
        return LabeledDatasIterator(inputs, outputs);
    }

    LabeledDatasIterator end() const {
        return LabeledDatasIterator(inputs, outputs, nSamples);
    }

    size_t size() const { return nSamples; }
    size_t inputDim() const { return inputDimension; }
    size_t outputDim() const { return outputDimension; }

    LabeledSet &append(Input &input, Output &output) {
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

    LabeledSet &append(const Input &input, const Output &output) {
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

    LabeledSet &append(LabeledData &sample) {
        return append(sample.data, sample.label);
    }

    LabeledSet &append(const LabeledData &sample) {
        return append(sample.data, sample.label);
    }

    friend std::istream &operator>>(std::istream &out, LabeledSet &ls);
    friend std::ostream &operator<<(std::ostream &out, const LabeledSet &ls);
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


/** `LabeledSet`'s input format is compatible with FANN library. */
std::istream &operator>>(std::istream &in, LabeledSet &ls) {
    ls.readFANN(in);
    return in;
}

/** `LabeledSet`'s output format is compatible with FANN library. */
std::ostream &operator<<(std::ostream &out, const LabeledSet &ls) {
    out << ls.nSamples << " "
        << ls.inputDimension << " "
        << ls.outputDimension << "\n";
    for (auto sample : ls) {
        out << sample << "\n";
    }
    return out;
}

#endif /* DATASET_H */

