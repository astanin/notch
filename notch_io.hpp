#ifndef NOTCH_IO_H
#define NOTCH_IO_H

/// notch_io.hpp -- optional input-output hepers for Notch library

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

#include <fstream>    // ifstream
#include <istream>
#include <ostream>

#include "notch.hpp"


/**
 * Vectors Input-Output
 * --------------------
 **/

/** Input and Output values are space-separated values.*/
std::istream &operator>>(std::istream &in, Input &xs) {
    for (size_t i = 0; i < xs.size(); ++i) {
        in >> xs[i];
    }
    return in;
}

std::ostream &operator<<(std::ostream &out, const Input &xs) {
    for (auto it = std::begin(xs); it != std::end(xs); ++it) {
        if (it != std::begin(xs)) {
            out << " ";
        }
        out << *it;
    }
    return out;
}


/**
 * Dataset Input-output
 * --------------------
 **/

/** Load labeled datasets from FANN text file format.
 *
 * N_samples N_in N_out
 * X[0,0] X[0,1] ... X[0,N_in - 1]
 * Y[0,0] Y[0,1] ... Y[0,N_out - 1]
 * X[1,0] X[1,1] ... X[1,N_in - 1]
 * Y[1,0] Y[1,1] ... Y[1,N_out - 1]
 * ...
 * X[N_samples - 1,0] X[N_samples - 1,1] ... X[N_samples - 1,N_in - 1]
 * Y[N_samples - 1,0] Y[N_samples - 1,1] ... Y[N_samples - 1,N_out - 1]
 **/
class FANNReader {
public:
    static LabeledDataset read(const std::string &path) {
        std::ifstream in(path);
        return FANNReader::read(in);
    }

    static LabeledDataset read(std::istream &in) {
        LabeledDataset ds;
        size_t nSamples, inputDimension, outputDimension;
        in >> nSamples >> inputDimension >> outputDimension;
        for (size_t i = 0; i < nSamples; ++i) {
            Input input(inputDimension);
            Output output(outputDimension);
            in >> input >> output;
            ds.append(input, output);
        }
        return ds;
    }
};

// TODO: add *Writer classes
/** Labeled pairs are split in two lines. */
std::ostream &operator<<(std::ostream &out, const LabeledData &p) {
    out << p.data << "\n" << p.label;
    return out;
}

/** `LabeledDataset`'s output format is compatible with FANN library. */
std::ostream &operator<<(std::ostream &out, const LabeledDataset &ls) {
    out << ls.size() << " "
        << ls.inputDim() << " "
        << ls.outputDim() << "\n";
    for (auto sample : ls) {
        out << sample << "\n";
    }
    return out;
}


/**
 * Neural Networks Input-Output
 * ----------------------------
 **/

std::ostream &operator<<(std::ostream &out, const ActivationFunction &af) {
    af.print(out);
    return out;
}

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

#endif
