#ifndef DATASET_H
#define DATASET_H


#include <sstream>
#include <vector>


using namespace std;


class LabeledSet {
    public:
        int nSamples;
        int inputSize;
        int outputSize;

        vector< vector<double> > inputs;
        vector< vector<double> > outputs;

        LabeledSet() {}
        LabeledSet(istream& in) { load(in); }

        /// Load a dataset from file (same format as FANN)
        void load(istream& in) {
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

        string fmt() {
            ostringstream ss;
            ss << "nSamples: " << nSamples << "\n";
            for (int i=0; i<nSamples; ++i) {
                for (int j=0; j<inputSize; ++j) {
                    ss << inputs[i][j] << " ";
                }
                ss << "=>";
                for (int j=0; j<outputSize; ++j) {
                    ss << " " << outputs[i][j];
                }
                ss << "\n";
            }
            return ss.str();
        }
};

#endif /* DATASET_H */

