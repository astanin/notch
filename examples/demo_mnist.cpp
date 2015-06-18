#include <iostream>
#include <fstream>


#include "notch.hpp"
#include "notch_io.hpp"


using namespace std;


int main() {
    ifstream trainImages("../data/train-images-idx3-ubyte", ios_base::binary);
    if (!trainImages.is_open()) {
        cerr << "Can't open file ../data/train-images-idx3-ubyte:\n"
             << "Run getmnist.py in ../data/ to download data\n";
        exit(0);
    }
    ifstream trainLabels("../data/train-labels-idx1-ubyte", ios_base::binary);
    if (!trainLabels.is_open()) {
        cerr << "Can't open file ../data/train-labels-idx1-ubyte:\n"
             << "Run getmnist.py in ../data/ to download data\n";
        exit(0);
    }
    LabeledDataset mnist = IDXReader(trainImages, trainLabels).read();
}
