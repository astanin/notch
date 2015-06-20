#ifndef NOTCH_IO_H
#define NOTCH_IO_H

/// @file notch_io.hpp optional input-output helpers

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

#include <algorithm>  // find_if
#include <fstream>    // ifstream
#include <istream>
#include <map>
#include <ostream>
#include <sstream>    // ostringstream
#include <stdexcept>  // out_of_range

#include "notch.hpp"


/* Vectors Input-Output
 * --------------------
 */

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


/** Load labeled datasets from CSV files.
 *
 * Numeric columns are converted to `float` numbers.
 *
 * Values of non-numeric columns are converted to numbers in range [0;N-1],
 * where N is the number of unique values in the column.
 *
 * By default the last column is used as a label (`labelcols={-1}`).
 **/
class CSVReader {
private:
    const char delimiter = ',';
    std::istream *inPtr;

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
    std::vector<std::string> readCSVRow(const std::string &row) {
        CSVReaderState state = CSVReaderState::UnquotedField;
        std::vector<std::string> fields {""};
        size_t i = 0; // index of the current field
        for (char c : row) {
            switch (state) {
                case CSVReaderState::UnquotedField:
                    if (c == delimiter) { // end of field
                          fields.push_back(""); i++;
                    } else if (c == '"') {
                        state = CSVReaderState::QuotedField;
                    } else {
                        fields[i].push_back(c);
                    }
                    break;
                case CSVReaderState::QuotedField:
                    if (c == '"') {
                        state = CSVReaderState::QuotedQuote;
                    } else {
                        fields[i].push_back(c);
                    }
                    break;
                case CSVReaderState::QuotedQuote:
                    if (c == delimiter) { // , after closing quote
                          fields.push_back(""); i++;
                          state = CSVReaderState::UnquotedField;
                    } else if (c == '"') { // "" -> "
                          fields[i].push_back('"');
                          state = CSVReaderState::QuotedField;
                    } else { // end of quote
                          state = CSVReaderState::UnquotedField;
                    }
                    break;
            }
        }
        return fields;
    }

    /// Read Excel CSV file. Skip empty rows.
    TextTable readCSV(std::istream &in, int skiprows=0, int skipcols=0) {
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

    Cell parseCell(const std::string &s) {
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

    MixedTable convertToMixed(const TextTable &table) {
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

    bool
    isColumnNumeric(const MixedTable &t, size_t column_idx) {
        return std::all_of(std::begin(t), std::end(t),
                [column_idx](const MixedRow &r) {
                    return r.at(column_idx).tag == CellTag::Number;
                });
    }

    NumericTable
    convertToNumeric(const MixedTable &t) {
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
    /** Construct a CSVReader. */
    CSVReader(const char delimiter=',')
        : delimiter(delimiter) {}
    /** Construct a CSVReader and set its default input stream. */
    CSVReader(std::istream &in, const char delimiter=',')
        : delimiter(delimiter), inPtr(&in) {}

    /** Read a `LabeledDataset` from the default input stream.
     *
     * @param labelcols  indices of the columns to be used as labels;
     *                   indices can be negative (-1 is the last column)
     * @param skiprows   discard the first @skiprows lines
     * @param skipcols   discard the first @skipcols columns
     **/
    LabeledDataset
    read(std::vector<int> labelcols = {-1}, int skiprows=0, int skipcols=0) {
        if (!inPtr) {
            throw std::logic_error("CSVReader has no input stream");
        }
        return CSVReader::read(*inPtr, labelcols, skiprows, skipcols);
    }

    /** Read a `LabeledDataset` from a CSV file.
     *
     * @param path       CSV file name
     * @param labelcols  indices of the columns to be used as labels;
     *                   indices can be negative (-1 is the last column)
     * @param skiprows   discard the first @skiprows lines
     * @param skipcols   discard the first @skipcols columns
     **/
    LabeledDataset
    read(const std::string &path, std::vector<int> labelcols = {-1},
         int skiprows=0, int skipcols=0) {
        std::ifstream in(path);
        return CSVReader::read(in, labelcols, skiprows, skipcols);
    }

    /** Read a `LabeledDataset` from an `std::istream`.
     *
     * @param in         stream to read CSV data from
     * @param labelcols  indices of the columns to be used as labels;
     *                   indices can be negative (-1 is the last column)
     * @param skiprows   discard the first @skiprows lines
     * @param skipcols   discard the first @skipcols columns
     **/
    LabeledDataset
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

/** Read datasets (images) from IDX File Format (as in MNIST dataset).
 *
 * References:
 *
 *  - http://yann.lecun.com/exdb/mnist/
 */
class IDXReader {
protected:
    std::istream *imagesInPtr = nullptr;
    std::istream *labelsInPtr = nullptr;

    static const int maxDims = 4;
    size_t imagesDims = 0;
    size_t labelsDims = 0;
    size_t imagesShape[IDXReader::maxDims] = {0, 0, 0, 0};
    size_t labelsShape[IDXReader::maxDims] = {0, 0, 0, 0};

    Dataset readDataset(std::istream &in, size_t *dims, size_t shape[]) {
        Dataset dataset;
        // read shape information
        *dims = readDims(in);
        readShape(in, *dims, shape);
        // calculate sample size in bytes
        size_t nSamples = shape[0];
        size_t nSampleSize = sizeof(uint8_t);
        for (size_t i = 1; i < *dims; ++i) {
            nSampleSize *= shape[i];
        }
        // read all samples
        dataset.reserve(nSamples);
        std::vector<uint8_t> buf(nSampleSize, 0);
        for (size_t i = 0; i < nSamples; ++i) {
            Array sample(0.0, nSampleSize);
            in.read((char*)buf.data(), nSampleSize);
            size_t n = in.gcount();
            if (n != nSampleSize) {
                throw std::runtime_error("IDXReader: cannot read data sample");
            }
            for (size_t j = 0; j < nSampleSize; ++j) {
                sample[j] = buf[j]; // convert uint8_t to float
            }
            dataset.push_back(sample);
        }
        return dataset;
    }

    size_t readDims(std::istream &in) {
        char magic[4] = { 0, 0, 0, 0};
        in.read(magic, 4);
        size_t n = in.gcount();
        for (size_t i = 0; i<4; ++i)
        if (n != 4) {
            throw std::runtime_error("IDXReader cannot read magic");
        }
        if (magic[0] != 0 || magic[1] != 0) {
            throw std::runtime_error("IDXReader: not an IDX magic");
        }
        char typecode = magic[2];
        if (typecode != 0x08) { // unsigned byte
            throw std::runtime_error("IDXReader: unsupported IDX type code");
        }
        char dims = magic[3];
        if (dims > maxDims || dims <= 0) {
            throw std::runtime_error("IDXReader: unsupported number of dimensions");
        }
        return dims;
    }

    void readShape(std::istream &in, size_t dims, size_t shape[]) {
        for (size_t i = 0; i < dims; ++i) {
            shape[i] = readBigEndian(in);
        }
    }

    uint32_t readBigEndian(std::istream &in) { // ntohl is less portable
        uint32_t x = 0;
        uint8_t buf[4] = {0, 0, 0, 0};
        in.read((char*) buf, 4);
        size_t n = in.gcount();
        if (n != 4) {
            throw std::runtime_error("IDXReader: cannot read a 4-byte integer");
        }
        x |= (uint32_t(buf[0]) << 24);
        x |= (uint32_t(buf[1]) << 16);
        x |= (uint32_t(buf[2]) << 8);
        x |= (uint32_t(buf[3]));
        return x;
    }

public:
    /** Construct an IDXReader. */
    IDXReader() {}
    /** Construct an IDXReader and set its default input streams (images and labels). */
    IDXReader(std::istream &imagesIn, std::istream &labelsIn)
        : imagesInPtr(&imagesIn), labelsInPtr(&labelsIn) {}

    /** Read a `LabeledDataset` from default input streams.  */
    LabeledDataset read() {
        if (!imagesInPtr || !labelsInPtr) {
            throw std::logic_error("IDXReader has no input stream");
        }
        return read(*imagesInPtr, *labelsInPtr);
    }

    /** Read a `LabeledDataset` from two files (images and labels respectively). */
    LabeledDataset read(const std::string &imagesFilename,
                        const std::string &labelsFilename) {
        std::ifstream imagesfile(imagesFilename, std::ios_base::binary);
        std::ifstream labelsfile(labelsFilename, std::ios_base::binary);
        return read(imagesfile, labelsfile);
    }

    /** Read a `LabeledDataset` from two streams (images and labels respectively). */
    LabeledDataset read(std::istream &imagesIn, std::istream &labelsIn) {
        LabeledDataset dataset;
        Dataset images = readDataset(imagesIn, &imagesDims, imagesShape);
        Dataset labels = readDataset(labelsIn, &labelsDims, labelsShape);
        if (labelsDims != 1) {
            throw std::runtime_error("IDXReader: labels dataset is not 1D");
        }
        if (imagesShape[0] != labelsShape[0]) {
            throw std::runtime_error("IDXReader: # of images and labels don't match");
        }
        size_t nSamples = imagesShape[0];
        for (size_t i = 0; i < nSamples; ++i) {
            dataset.append(images[i], labels[i]);
        }
        return dataset;
    }
};

/** A formatter to write a labeled dataset to CSV file. */
class CSVWriter {
private:
    std::ostream &out;
public:
    CSVWriter(std::ostream &out) : out(out) {}

    std::ostream &operator<<(LabeledDataset &dataset) {
        auto inDim = dataset.inputDim();
        auto outDim = dataset.outputDim();
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
        for (auto sample : dataset) {
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
};


/* Neural Networks Input-Output
 * ----------------------------
 */


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


/// Access protected activation function of LAYER classes.
template<class LAYER>
class GetActivation : public LAYER {
public:
   /// Get a reference to protected 'bias' member of a LAYER class.
    static const Activation& ref(const LAYER &l) {
        auto &access = static_cast<const GetActivation<LAYER>&>(l);
        return *access.activationFunction;
    }
};


/** Type-agnostic layer specification for serialization. */
struct LayerSpec {
    std::string tag; //< layer type name
    size_t inputDim = 0; //< the number of layer inputs
    size_t outputDim = 0; //< the number of layer outputs (the number of nodes)
    std::shared_ptr<std::string> activation = nullptr;
    std::shared_ptr<Array> weights = nullptr;
    std::shared_ptr<Array> bias = nullptr;

    LayerSpec(const ABackpropLayer &layer) {
        std::string tag = layer.tag();
        this->tag = tag;
        this->inputDim = layer.inputDim();
        this->outputDim = layer.outputDim();
        if (tag == "FullyConnectedLayer") {
            using LT = FullyConnectedLayer;
            auto &l = dynamic_cast<const LT&>(layer);
            auto &af = GetActivation<LT>::ref(l);
            auto &ws = GetWeights<LT>::ref(l);
            auto &bs = GetBias<LT>::ref(l);
            activation = std::make_shared<std::string>(af.tag());
            weights = std::make_shared<Array>(ws);
            bias = std::make_shared<Array>(bs);
        } else if (tag == "ActivationLayer") {
            using LT = FullyConnectedLayer;
            auto &l = dynamic_cast<const LT&>(layer);
            auto &af = GetActivation<LT>::ref(l);
            activation = std::make_shared<std::string>(af.tag());
        } else {
            throw std::invalid_argument("unsupported layer type: " + tag);
        }
    }

    LayerSpec(const ALossLayer &layer) {
        std::string tag = layer.tag();
        this->tag = tag;
        this->inputDim = layer.inputDim();
        this->outputDim = 0;
    }
};


/** Type-agnostic specification of the neural network for serialization. */
struct NetSpec {
    std::string tag; //< network type name
    size_t inputDim = 0; //< the number of inputs of the first layer
    size_t outputDim = 0; //< the number of outputs of the last layer
    std::vector<LayerSpec> layers; //< all layers, loss including
    bool hasLoss = false; //< true if there's a loss layer

    NetSpec(const Net &net) {
        this->tag = "Net";
        for (size_t i = 0; i < net.size(); ++i) {
            auto layer = net.getLayer(i);
            if (layer) {
                auto spec = LayerSpec(*layer);
                layers.push_back(spec);
                if (i == 0) {
                    inputDim = spec.inputDim;
                }
                outputDim =spec.outputDim;
            }
        }
        auto lossLayer = net.getLossLayer();
        if (lossLayer) {
            layers.push_back(LayerSpec(*lossLayer));
            hasLoss = true;
        }
    }
};


/// Read neural network parameters from a record-jar text file.
///
/// See http://catb.org/~esr/writings/taoup/html/ch05s02.html#id2906931
class PlainTextNetworkReader {
private:
    std::istream *inPtr = nullptr;

    const std::map<std::string, const Activation&>
         knownActivations = {{"tanh", defaultTanh},
                             {"scaledTanh", scaledTanh},
                             {"linear", linearActivation},
                             {"ReLU", ReLU},
                             {"leakyReLU", leakyReLU}};

    std::string read_tag(std::istream &in) {
        std::string tag;
        in >> std::ws >> tag >> std::ws;
        return tag;
    }

    template<typename T> T
    read_value(std::istream &in, T &value) {
       in >> std::ws >> value >> std::ws;
       return value;
    }

    /// Read a table of bias_and_weights.
    ///
    /// Table format:
    ///
    ///     b_0 w_00 w_01 w_02 ...
    ///     b_1 w_10 w_11 w_12 ...
    ///     ...
    ///
    void read_weights(std::istream &in, Array &w, Array &bias) {
        size_t nInputs = w.size() / bias.size();
        size_t nOutputs = bias.size();
        for (size_t row = 0; row < nOutputs; ++row) {
            in >> std::ws >> bias[row];
            for (size_t col = 0; col < nInputs; ++col) {
                in >> std::ws >> w[row*nInputs + col];
            }
        }
    }

    template<typename VALUE>
    void read_tag_value(std::istream &in, std::string const &tag, VALUE &val) {
        std::string inTag = read_tag(in);
        if (inTag != tag) {
            throw std::runtime_error("tag '" + tag + "' not found");
        }
        read_value<VALUE>(in, val);
    }

    /// read end-of-record sequence if there is any
    bool
    consume_end_of_record(std::istream &in) {
        bool isEOR = false;
        in >> std::ws;          // consume whitespace
        if (!in) {
            return false;
        }
        if (in.peek() == '%') {
           in.get();            // read the first %
           if (!in) {
               return false;
           }
           if (in.peek() == '%') {
               in.get();        // read also the second %
               in >> std::ws;   // consume whitespace
               isEOR = true;
           } else {
               in.putback('%'); // the first %
           }
        }
        return isEOR;
    }

    /// Read network header block.
    /// @return the number of layers.
    size_t read_header(std::istream &in, MakeNet &mknet) {
        std::string netTag;
        float fmtVersion;
        size_t inputDim;
        size_t outputDim;
        size_t nLayers;
        read_tag_value<std::string>(in, "net:", netTag);
        if (netTag != "Net") {
            throw std::runtime_error("unsupported network type: " + netTag);
        }
        read_tag_value<float>(in, "format:", fmtVersion);
        if (fmtVersion != 1.0) {
            throw std::runtime_error("unsupported version network format");
        }
        read_tag_value<size_t>(in, "inputs:", inputDim);
        read_tag_value<size_t>(in, "outputs:", outputDim); // ignore
        read_tag_value<size_t>(in, "layers:", nLayers);
        consume_end_of_record(in);
        mknet.setInputDim(inputDim);
        return nLayers;
    }

    void read_layer_config(std::istream &in,
                           size_t &inputDim, size_t &outputDim,
                           std::string &activationTag,
                           Array &weights, Array &bias) {
        std::string tag;
        while (in && !in.eof()) {
            tag = read_tag(in);
            if (!in) { // trying to read past EOF or other errors
                throw std::runtime_error("unexpected end of layer record");
            }
            if (tag == "inputs:") {
                read_value<size_t>(in, inputDim);
            } else if (tag == "outputs:") {
                read_value<size_t>(in, outputDim);
            } else if (tag == "activation:") {
                read_value<std::string>(in, activationTag);
            } else if (tag == "bias_and_weights:") {
                weights.resize(inputDim * outputDim, 0.0);
                bias.resize(outputDim, 0.0);
                read_weights(in, weights, bias);
                break; // this should be the last layer attribute
            } else {
                throw std::runtime_error("unsupported layer attribute: " + tag);
            }
        };
    }

    void read_layer(std::istream &in, MakeNet &mknet) {
        std::string tag;
        size_t inputDim = 0;
        size_t outputDim = 0;
        std::string activationTag = "";
        Array w(0);
        Array b(0);
        read_tag_value<std::string>(in, "layer:", tag);
        read_layer_config(in, inputDim, outputDim, activationTag, w, b);
        if (tag == "FullyConnectedLayer") {
            auto af = knownActivations.find(activationTag);
            if (af != knownActivations.end()) {
                mknet.addFC(w, b, af->second);
            } else {
                throw std::runtime_error("unsupported activation: " +activationTag);
            }
        } else if (tag == "ActivationLayer") {
            auto af = knownActivations.find(activationTag);
            if (af != knownActivations.end()) {
                mknet.addActivation(af->second);
            } else {
                throw std::runtime_error("unsupported activation: " +activationTag);
            }
        } else if (tag == "EuclideanLoss") {
            mknet.addL2Loss();
        } else if (tag == "HingeLoss") {
            mknet.addHingeLoss();
        } else if (tag == "SoftmaxWithLoss") {
            mknet.addSoftmax();
        } else {
            throw std::runtime_error("unsupported layer type: " + tag);
        }
    }

    MakeNet &read_net(std::istream &in, MakeNet &mknet) {
        size_t nLayers;
        nLayers = read_header(in, mknet);
        consume_end_of_record(in);
        for (size_t i = 0; i < nLayers; ++i) {
            read_layer(in, mknet);
            consume_end_of_record(in);
        }
        return mknet;
    }

public:

    /** Construct a PlainTextNetworkReader. */
    PlainTextNetworkReader() {}
    /** Construct a PlainTextNetworkReader and set its default input stream. */
    PlainTextNetworkReader(std::istream &in) : inPtr(&in) {}

    /** Read a feed-forward `Net` from the default stream. */
    Net read() {
        MakeNet mknet;
        if (!inPtr) {
            throw std::logic_error("PlainTextNetworkReader has no input stream");
        }
        return read_net(*inPtr, mknet).make();
    }

    /** Read a feed-forward `Net` from file. */
    Net read(const std::string &netFilename) {
        MakeNet mknet;
        std::ifstream netfile(netFilename);
        return read_net(netfile, mknet).make();
    }

    /** Read a feed-forward `Net` from stream. */
    Net read(std::istream &in) {
        MakeNet mknet;
        return read_net(in, mknet).make();
    }
};


/// Write neural network parameters to a record-jar text file.
///
/// See http://catb.org/~esr/writings/taoup/html/ch05s02.html#id2906931
class PlainTextNetworkWriter {
private:
    std::ostream &out;

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

public:
    PlainTextNetworkWriter(std::ostream &out) : out(out) {}

    void save(LayerSpec &spec) {
        out << "layer: " << spec.tag << "\n";
        if (spec.inputDim) {
            out << "inputs: " << spec.inputDim << "\n";
        }
        if (spec.outputDim) {
            out << "outputs: " << spec.outputDim << "\n";
        }
        if (spec.activation) {
            out << "activation: " << *spec.activation << "\n";
        }
        if (spec.weights && spec.bias) {
            out << "bias_and_weights:\n";
            saveWeightsAndBias(*spec.weights, *spec.bias);
        }
    }

    template<class LAYER>
    void save(const LAYER &layer) {
        auto spec = LayerSpec(layer);
        save(spec);
    }

    void save(const Net &net) {
        auto netSpec = NetSpec(net);
        size_t n = netSpec.layers.size();
        out << "net: " << netSpec.tag << "\n";
        out << "format: 1.0\n"; // format version
        out << "inputs: " << netSpec.inputDim << "\n";
        out << "outputs: " << netSpec.outputDim << "\n";
        out << "layers: " << n << "\n";
        out << "%%\n";
        for (size_t i = 0; i < n; ++i) {
            save(netSpec.layers[i]);
            if (i < (n - 1)) {
                out << "%%\n";
            }
        }
    }

    PlainTextNetworkWriter &operator<<(const std::string &s) {
        out << s;
        return *this;
    }

    PlainTextNetworkWriter &operator<<(const ABackpropLayer &layer) {
        save<ABackpropLayer>(layer);
        return *this;
    }

    PlainTextNetworkWriter &operator<<(const ALossLayer &layer) {
        save<ALossLayer>(layer);
        return *this;
    }

    PlainTextNetworkWriter &operator<<(const Net &net) {
        save(net);
        return *this;
    }
};

#endif
