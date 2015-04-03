#ifndef DATASET_H
#define DATASET_H


#include <sstream>
#include <vector>
#include <iterator>


using namespace std;


using Input = vector<double>;
using Output = vector<double>;


struct LabeledPair {
    Input const &input;
    Output const &output;

    LabeledPair(const Input& input, const Output& output) :
        input(input), output(output) {}

    string fmt() {
        ostringstream ss;
        for (auto x : input) {
            ss << x << " ";
        }
        ss << "=>";
        for (auto y : output) {
            ss << " " << y;
        }
        return ss.str();
    }
};


class LabeledPairsIterator : public iterator<input_iterator_tag, LabeledPair> {
    private:
        vector<Input> const *inputs;
        vector<Output> const *outputs;
        size_t position;

    public:
        LabeledPairsIterator
            (const vector<Input> &inputs,
             const vector<Output> &outputs,
             size_t position = 0) :
            inputs(&inputs), outputs(&outputs), position(position) {};

        bool operator==(const LabeledPairsIterator& rhs) const {
            return (inputs == rhs.inputs &&
                    outputs == rhs.outputs &&
                    position == rhs.position);
        }

        bool operator!=(const LabeledPairsIterator& rhs) const {
            return !(*this == rhs);
        }

        LabeledPair operator*() const {
            LabeledPair lp(inputs->at(position), outputs->at(position));
            return lp;
        }

        LabeledPairsIterator& operator++() {
            if (position < inputs->size()) {
                ++position;
            }
            return *this;
        }

        LabeledPairsIterator& operator++(int) {
            return ++(*this);
        }
};


class LabeledSet {
    private:
        /// load a dataset from file (the same format as FANN)
        void loadFANN(istream& in) {
            in >> nSamples >> inputSize >> outputSize;
            inputs.clear();
            outputs.clear();
            for (int i=0; i<nSamples; ++i) {
                auto input = vector<double>(inputSize);
                auto output = vector<double>(outputSize);
                for (int j=0; j<inputSize; ++j) {
                    in >> input[j];
                }
                for (int j=0; j<outputSize; ++j) {
                    in >> output[j];
                }
                inputs.push_back(input);
                outputs.push_back(output);
            }
        }

        /// format dataset as FANN plain-text format
        string fmtFANN() {
            ostringstream ss;
            ss << nSamples << " " << inputSize << " " << outputSize << "\n";
            for (int i=0; i<nSamples; ++i) {
                for (int j=0; j<inputSize; ++j) {
                    if (j) {
                        ss << " ";
                    }
                    ss << inputs[i][j];
                }
                ss << "\n";
                for (int j=0; j<outputSize; ++j) {
                    if (j) {
                        ss << " ";
                    }
                    ss << outputs[i][j];
                }
                ss << "\n";
            }
            return ss.str();
        }

   public:
        int nSamples;
        int inputSize;
        int outputSize;

        vector< Input > inputs;
        vector< Output > outputs;

        LabeledSet() {}
        LabeledSet(istream& in) { loadFANN(in); }

        LabeledPairsIterator begin() {
            return LabeledPairsIterator(inputs, outputs);
        }

        LabeledPairsIterator end() {
            return LabeledPairsIterator(inputs, outputs, nSamples);
        }

        friend istream& operator>>(istream& out, LabeledSet& ls);
        friend ostream& operator<<(ostream& out, LabeledSet& ls);
};

istream& operator>>(istream& in, LabeledSet& ls) {
    ls.loadFANN(in);
    // if (newLS.isBAD(TODO)) in.setstate(ios::failbit);
    return in;
}

ostream& operator<<(ostream& out, LabeledSet& ls) {
    out << ls.fmtFANN();
    return out;
}

#endif /* DATASET_H */

