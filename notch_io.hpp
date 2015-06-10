#ifndef NOTCH_IO_H
#define NOTCH_IO_H

/// notch_io.hpp -- optional input-output helpers for Notch library

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

#include <algorithm>  // find_if
#include <fstream>    // ifstream
#include <istream>
#include <map>
#include <ostream>
#include <sstream>    // ostringstream
#include <stdexcept>  // out_of_range

#include "notch.hpp"


/**
 * Vectors Input-Output
 * --------------------
 **/

std::istream &operator>>(std::istream &in, Array &xs);
std::ostream &operator<<(std::ostream &out, const Array &xs);
std::ostream &operator<<(std::ostream &out, const std::valarray<double> &xs);

#ifndef NOTCH_ONLY_DECLARATIONS
/** Input and Output values are space-separated values.*/
std::istream &operator>>(std::istream &in, Array &xs) {
    for (size_t i = 0; i < xs.size(); ++i) {
        in >> xs[i];
    }
    return in;
}

std::ostream &operator<<(std::ostream &out, const Array &xs) {
    for (auto it = std::begin(xs); it != std::end(xs); ++it) {
        if (it != std::begin(xs)) {
            out << " ";
        }
        out << *it;
    }
    return out;
}

std::ostream &operator<<(std::ostream &out, const std::valarray<double> &xs) {
    for (auto it = std::begin(xs); it != std::end(xs); ++it) {
        if (it != std::begin(xs)) {
            out << " ";
        }
        out << *it;
    }
    return out;
}
#endif


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
        if (!in.is_open()) {
            throw std::runtime_error("cannot open " + path);
        }
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

/** Load labeled datasets from CSV files.
 *
 * Numeric columns are converted to `float` numbers.
 *
 * Values of non-numeric columns are converted to numbers in range [0;N-1],
 * where N is the number of unique values in the column.
 *
 * By default the last column is used as a label (`labelcols={-1}`).
 **/
template<char delimiter=','>
class CSVReader {
private:
    enum class CellTag { Number, String };

    struct Cell {
        CellTag tag;
        float value;
        std::string str;
    };

    using TextTable = std::vector<std::vector<std::string>>;
    using MixedTable = std::vector<std::vector<Cell>>;
    using MixedRow = std::vector<Cell>;
    using NumericTable = std::vector<std::vector<float>>;
    using NumericRow = std::vector<float>;

    enum class CSVReaderState {
        UnquotedField,
        QuotedField,
        QuotedQuote
    };

    /// Parse a row of an Excel CSV file.
    static std::vector<std::string>
    readCSVRow(const std::string &row) {
        CSVReaderState state = CSVReaderState::UnquotedField;
        std::vector<std::string> fields {""};
        size_t i = 0; // index of the current field
        for (char c : row) {
            switch (state) {
                case CSVReaderState::UnquotedField:
                    switch (c) {
                        case delimiter: // end of field
                                  fields.push_back(""); i++;
                                  break;
                        case '"': state = CSVReaderState::QuotedField;
                                  break;
                        default:  fields[i].push_back(c);
                                  break; }
                    break;
                case CSVReaderState::QuotedField:
                    switch (c) {
                        case '"': state = CSVReaderState::QuotedQuote;
                                  break;
                        default:  fields[i].push_back(c);
                                  break; }
                    break;
                case CSVReaderState::QuotedQuote:
                    switch (c) {
                        case delimiter: // , after closing quote
                                  fields.push_back(""); i++;
                                  state = CSVReaderState::UnquotedField;
                                  break;
                        case '"': // "" -> "
                                  fields[i].push_back('"');
                                  state = CSVReaderState::QuotedField;
                                  break;
                        default:  // end of quote
                                  state = CSVReaderState::UnquotedField;
                                  break; }
                    break;
            }
        }
        return fields;
    }

    /// Read Excel CSV file. Skip empty rows.
    static TextTable
    readCSV(std::istream &in, int skiprows=0, int skipcols=0) {
        TextTable table;
        std::string row;
        while (true) {
            std::getline(in, row);
            if (in.bad() || in.eof()) {
                break;
            }
            if (skiprows > 0) {
                --skiprows;
                continue;
            }
            if (row.size() == 0) {
                continue;
            }
            auto fields = readCSVRow(row);
            if (skipcols > 0) {
                auto field0 = std::begin(fields);
                auto fieldToKeep = field0 + skipcols;
                fields.erase(field0, fieldToKeep);
            }
            table.push_back(fields);
        }
        return table;
    }

    static Cell
    parseCell(const std::string &s) {
        // float value = std::stof(s); // doesn't work on MinGW-32; BUG #52015
        const char *parseBegin = s.c_str();
        char *parseEnd = nullptr;
        float value = strtof(parseBegin, &parseEnd);
        if (parseEnd == parseBegin || parseEnd == nullptr) {
            return Cell { CellTag::String, 0.0, s };
        } else {
            return Cell { CellTag::Number, value, "" };
        }
    }

    static MixedTable
    convertToMixed(const TextTable &table) {
       MixedTable cellTable(0);
       for (auto row : table) {
           MixedRow cellRow(0);
           for (auto s : row) {
               cellRow.push_back(parseCell(s));
           }
           cellTable.push_back(cellRow);
       }
       return cellTable;
    }

    static bool
    isColumnNumeric(const MixedTable &t, size_t column_idx) throw (std::out_of_range) {
        return std::all_of(std::begin(t), std::end(t),
                [column_idx](const MixedRow &r) {
                    return r.at(column_idx).tag == CellTag::Number;
                });
    }

    ///
    static NumericTable
    convertToNumeric(const MixedTable &t) throw (std::out_of_range) {
        // analyze table
        size_t ncols = t.at(0).size();
        std::vector<bool> isNumeric(ncols);
        for (size_t i = 0; i < ncols; ++i) {
            isNumeric[i] = isColumnNumeric(t, i);
        }
        // convert table
        std::vector<std::map<std::string, int>> columnKeys(ncols);
        NumericTable nt;
        for (auto row : t) {
            NumericRow nr;
            for (size_t i = 0; i < ncols; ++i) {
                if (isNumeric[i]) {
                    nr.push_back(row[i].value);
                } else { // a non-numeric label
                    auto label = row[i].str;
                    if (columnKeys[i].count(label)) { // a previosly seen label
                        int labelIndex = columnKeys[i][label];
                        nr.push_back(labelIndex);
                    } else { // new label
                        int labelIndex = columnKeys[i].size();
                        columnKeys[i][label] = labelIndex;
                        nr.push_back(labelIndex);
                    }
                }
            }
            nt.push_back(nr);
        }
        return nt;
    }

public:
    /** Read a `LabeledDataset` from a CSV file.
     *
     * @param path       CSV file name
     * @param labelcols  indices of the columns to be used as labels;
     *                   indices can be negative (-1 is the last column)
     * @param skiprows   discard the first @skiprows lines
     * @param skipcols   discard the first @skipcols lines
     **/
    static LabeledDataset
    read(const std::string &path, std::vector<int> labelcols = {-1},
         int skiprows=0, int skipcols=0) {
        std::ifstream in(path);
        return CSVReader::read(in, skiprows, skipcols);
    }

    /** Read a `LabeledDataset` from an `std::istream`.
     *
     * @param in         stream to read CSV data from
     * @param labelcols  indices of the columns to be used as labels;
     *                   indices can be negative (-1 is the last column)
     * @param skiprows   discard the first @skiprows lines
     * @param skipcols   discard the first @skipcols lines
     **/
    static LabeledDataset
    read(std::istream &in, std::vector<int> labelcols = {-1},
         int skiprows=0, int skipcols=0) {
        LabeledDataset ds;
        auto rows = readCSV(in, skiprows, skipcols);
        auto mixedRows = convertToMixed(rows);
        auto numericRows = convertToNumeric(mixedRows);
        for (auto row : numericRows) {
            // build label vector
            Output label(labelcols.size());
            size_t ncols = row.size();
            for (size_t i = 0; i < labelcols.size(); ++i) {
                size_t colIdx = ((ncols + labelcols[i]) % ncols);
                label[i] = row[colIdx];
            }
            // remove label columns from the row (from the end)
            for (int i = row.size()-1; i >= 0; --i) {
                auto found = std::find_if(labelcols.begin(), labelcols.end(),
                             [=](int labelcol) {
                                return (labelcol + ncols) % ncols == size_t(i);
                             });
                if (found != labelcols.end()) {
                    row.erase(row.begin() + i);
                }
            }
            // build data vector
            Input data(row.size());
            for (size_t i = 0; i < row.size(); ++i) {
                data[i] = row[i];
            }
            ds.append(data, label);
        }
        return ds;
    }
};

// TODO: IDX (MNIST) format reader
// TODO: LIBSVM format reader

/** A formatter to write a labeled dataset to FANN text file format. */
class FANNFormat {
private:
    const LabeledDataset &dataset;
public:
    FANNFormat(const LabeledDataset &dataset) : dataset(dataset) {}
    friend std::ostream &operator<<(std::ostream &out, const FANNFormat &w);
};

std::ostream &operator<<(std::ostream &out, const FANNFormat &w);
#ifndef NOTCH_ONLY_DECLARATIONS
std::ostream &operator<<(std::ostream &out, const FANNFormat &w) {
    out << w.dataset.size() << " "
        << w.dataset.inputDim() << " "
        << w.dataset.outputDim() << "\n";
    for (auto sample : w.dataset) {
        out << sample.data << "\n" << sample.label << "\n";
    }
    return out;
}
#endif

/** A formatter to write datasets as one mapping per line. */
class ArrowFormat {
private:
    const LabeledDataset &dataset;
public:
    ArrowFormat(const LabeledDataset &dataset) : dataset(dataset) {}
    friend std::ostream &operator<<(std::ostream &out, const ArrowFormat &w);
};

std::ostream &operator<<(std::ostream &out, const ArrowFormat &w);
#ifndef NOTCH_ONLY_DECLARATIONS
std::ostream &operator<<(std::ostream &out, const ArrowFormat &w) {
    for (auto sample : w.dataset) {
        out << sample.data << " -> " << sample.label << "\n";
    }
    return out;
}
#endif

/** A formatter to write a labeled dataset to CSV file. */
class CSVFormat {
private:
    const LabeledDataset &dataset;
public:
    CSVFormat(const LabeledDataset &dataset) : dataset(dataset) {}
    friend std::ostream &operator<<(std::ostream &out, const CSVFormat &w);
};

std::ostream &operator<<(std::ostream &out, const CSVFormat &writer);
#ifndef NOTCH_ONLY_DECLARATIONS
std::ostream &operator<<(std::ostream &out, const CSVFormat &writer) {
    auto inDim = writer.dataset.inputDim();
    auto outDim = writer.dataset.outputDim();
    auto w = 11;
    auto p = 5;
    // header row
    for (auto i = 1u; i <= inDim; ++i) {
        std::ostringstream ss;
        ss << "feature_" << i;
        out << std::setw(w) << ss.str() << ",";
    }
    for (auto i = 1u; i <= outDim; ++i) {
        std::ostringstream ss;
        ss << "label_" << i;
        out << std::setw(w) << ss.str();
        if (i < outDim) {
           out << ",";
        }
    }
    out << "\n";
    // data rows
    for (auto sample : writer.dataset) {
        for (auto v : sample.data) {
           out << std::setw(w) << std::setprecision(p) << v << ",";
        }
        for (auto i = 0u; i < outDim; ++i) {
           out << std::setw(w) << std::setprecision(p) << sample.label[i];
           if (i + 1 < outDim) {
               out << ",";
           }
        }
        out << "\n";
    }
    return out;
}
#endif

/**
 * Neural Networks Input-Output
 * ----------------------------
 **/


/// Access protected weights member of LAYER classes.
template<class LAYER>
class GetWeights : public LAYER {
public:
    /// Get a reference to protected 'weights' member of a LAYER class.
    static Array& ref(LAYER &l) {
        auto &access = static_cast<GetWeights<LAYER>&>(l);
        return access.weights;
    }
    /// Get a reference to protected 'weights' member of a LAYER class.
    static const Array& ref(const LAYER &l) {
        auto &access = static_cast<const GetWeights<LAYER>&>(l);
        return access.weights;
    }
};

/// Access protected bias member of LAYER classes.
template<class LAYER>
class GetBias : public LAYER {
public:
     /// Get a reference to protected 'bias' member of a LAYER class.
    static Array& ref(LAYER &l) {
        auto &access = static_cast<GetBias<LAYER>&>(l);
        return access.bias;
    }
    /// Get a reference to protected 'bias' member of a LAYER class.
    static const Array& ref(const LAYER &l) {
        auto &access = static_cast<const GetBias<LAYER>&>(l);
        return access.bias;
    }
};


/// Read neural network parameters from a record-jar text file.
///
/// See http://catb.org/~esr/writings/taoup/html/ch05s02.html#id2906931
#if 0
class PlainTextNetworkReader {
private:
    std::istream &in;

    const std::map<std::string, const Activation&>
         knownActivations = {{"tanh", defaultTanh},
                             {"scaledTanh", scaledTanh},
                             {"linear", linearActivation},
                             {"ReLU", ReLU},
                             {"leakyReLU", leakyReLU}};

    template<typename T> T
    read_tag_value(const std::string expected_tag, T &value) {
        std::string tag;
        in >> std::ws >> tag;
        if (tag != expected_tag) {
            auto what = "unexpected tag: " + tag + " != " + expected_tag;
            throw std::runtime_error(what);
        } else {
            in >> std::ws >> value;
        }
        return value;
    }

    void
    read_weights(const std::string expected_tag, Array &w, Array &bias) {
        size_t nInputs = w.size() / bias.size();
        size_t nOutputs = bias.size();
        std::string tag;
        in >> std::ws >> tag;
        if (tag != expected_tag) {
            auto what = "unexpected tag: " + tag + " != " + expected_tag;
            throw std::runtime_error(what);
        }
        for (size_t row = 0; row < nOutputs; ++row) {
            in >> std::ws >> bias[row];
            for (size_t col = 0; col < nInputs; ++col) {
                in >> std::ws >> w[row*nInputs + col];
            }
        }
    }

    /// read end-of-record sequence if there is any
    void
    consume_end_of_record() {
        in >> std::ws;          // consume whitespace
        if (in.peek() == '%') {
           in.get();            // read the first %
           if (in.peek() == '%') {
               in.get();        // read also the second %
           } else {
               in.putback('%'); // the first %
           }
        }
        in >> std::ws;          // consume whitespace
    }

public:

    PlainTextNetworkReader(std::istream &in = std::cin) : in(in) {}

    ABackpropLayer &load(FullyConnectedLayer &layer) {
        std::string layerTag, activationTag;
        size_t nInputs = 0;
        size_t nOutputs = 0;
        // read parameters
        read_tag_value("layer:", layerTag);
        if (layerTag != "FullyConnectedLayer") {
            auto what = "unsupported layer type: " + layerTag;
            throw std::runtime_error(what);
        }
        read_tag_value("inputs:", nInputs);
        read_tag_value("outputs:", nOutputs);
        read_tag_value("activation:", activationTag);
        if (! knownActivations.count(activationTag) ) {
            auto what = "unknown activation: " + activationTag;
            throw std::runtime_error(what);
        }
        const Activation &a = knownActivations.find(activationTag)->second;
        Array w(0.0, nInputs * nOutputs);
        Array b(0.0, nOutputs);
        read_weights("bias_and_weights:", w, b);
        consume_end_of_record();
        // modify the layer
        FCLParams::init(layer, nInputs, nOutputs, w, b);
        FCLParams::setActivation(layer, a);
        return layer;
    }

    MultilayerPerceptron &load(MultilayerPerceptron &mlp) {
        std::string netTag;
        size_t nLayers;
        read_tag_value("net:", netTag);
        if (netTag != "MultilayerPerceptron") {
            auto what = "unsupported network type: " + netTag;
            throw std::runtime_error(what);
        }
        read_tag_value("layers:", nLayers);
        consume_end_of_record();
        mlp.clear();
        for (size_t i = 0; i < nLayers; ++i) {
            auto layer = std::make_shared<FullyConnectedLayer>();
            load(*layer);
            mlp.append(std::move(layer));
        }
        return mlp;
    }

    PlainTextNetworkReader &operator>>(FullyConnectedLayer &layer) {
        load(layer);
        return *this;
    }

    PlainTextNetworkReader &operator>>(MultilayerPerceptron &mlp) {
        load(mlp);
        return *this;
    }
};
#endif

/// Write neural network parameters to a record-jar text file.
///
/// See http://catb.org/~esr/writings/taoup/html/ch05s02.html#id2906931
class PlainTextNetworkWriter {
private:
    std::ostream &out;

public:
    PlainTextNetworkWriter(std::ostream &out) : out(out) {}

    void saveWeightsAndBias(const Array &weights, const Array &bias) {
        size_t nOutputs = bias.size();
        size_t nInputs = weights.size() / nOutputs;
        for (size_t r = 0; r < nOutputs; ++r) {
            out << "   ";
            out << " " << bias[r];
            for (size_t c = 0; c < nInputs; ++c) {
                out << " " << weights[r*nInputs + c];
            }
            out << "\n";
        }
    }

    void save(const ABackpropLayer &layer) {
        out << "layer: " << layer.tag() << "\n";
        out << "inputs: " << layer.inputDim() << "\n";
        out << "outputs: " << layer.outputDim() << "\n";
        // TODO: print activation if any
        if (layer.tag() == "FullyConnectedLayer") { // if layer has parameters
            out << "bias_and_weights:\n";
            auto &fcl = dynamic_cast<const FullyConnectedLayer&>(layer);
            auto bias = GetBias<FullyConnectedLayer>::ref(fcl);
            auto weights = GetWeights<FullyConnectedLayer>::ref(fcl);
            saveWeightsAndBias(weights, bias);
        }
    }

    void save(const Net &net) {
        size_t nLayers = std::distance(net.begin(), net.end());
        if (!nLayers) {
            return;
        }
        out << "net: FeedForwardNet\n";
        out << "layers: " << nLayers << "\n";
        out << "%%\n";
        for (auto it = net.begin(); it != net.end(); ++it) {
            std::string tag = (**it).tag();
            auto &ref = (**it);
            save(ref);
            if (it + 1 != net.end()) {
                out << "%%\n";
            }
        }
    }

    PlainTextNetworkWriter &operator<<(const std::string &s) {
        out << s;
        return *this;
    }

    PlainTextNetworkWriter &operator<<(const FullyConnectedLayer &layer) {
        save(layer);
        return *this;
    }

    PlainTextNetworkWriter &operator<<(const Net &net) {
        save(net);
        return *this;
    }
};


#endif
