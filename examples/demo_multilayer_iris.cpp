/// usage: demo_multilayer_iris [path_to_data (default: ../data/iris.csv)]

#include <iostream>
#include <fstream>
#include <ostream>
#include <string>


#include "notch.hpp"
#include "notch_io.hpp"
#include "notch_pre.hpp"


using namespace std;


ostream &operator<<(ostream &out, const Dataset &d) {
    for (auto v : d) {
        out << v << "\n";
    }
    return out;
}


int main(int argc, char *argv[]) {
    string csvFile("../data/iris.csv");
    if (argc == 2) {
        csvFile = string(argv[1]);
    }
    cout << "reading dataset from CSV " << csvFile << "\n";
    ifstream f(csvFile);
    if (!f.is_open()) {
        cerr << "cannot open " << csvFile << "\n";
        exit(-1);
    }
    LabeledDataset ds = CSVReader<>::read(f);
    OneHotEncoder oh(ds.getOutputs());
    ds.transformLabels(oh);
    cout << ArrowFormat(ds);
}
