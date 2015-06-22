#include <iostream>
#include <fstream>
#include <string>


#include "notch.hpp"
#include "notch_io.hpp"      // IDXReader
#include "notch_pre.hpp"     // OneHotEncoder
#include "notch_metrics.hpp" // AClassifier


using namespace std;


const string IMAGES_FILE = "../data/train-images-idx3-ubyte";
const string LABELS_FILE = "../data/train-labels-idx1-ubyte";
const string IMAGES_TEST_FILE = "../data/t10k-images-idx3-ubyte";
const string LABELS_TEST_FILE = "../data/t10k-labels-idx1-ubyte";
const string SAVE_NETWORK_SNAPSHOT = "demo_mnist_network.txt";


LabeledDataset readMNIST(const string &imagesFile, const string labelsFile) {
    ifstream trainImages(imagesFile, ios_base::binary);
    if (!trainImages.is_open()) {
        cerr << "Can't open file " << imagesFile << "\n"
             << "Run getmnist.py in ../data/ to download data\n";
        exit(0);
    }
    ifstream trainLabels(labelsFile, ios_base::binary);
    if (!trainLabels.is_open()) {
        cerr << "Can't open file " << labelsFile << "\n"
             << "Run getmnist.py in ../data/ to download data\n";
        exit(0);
    }
    cout << "reading MNIST images from " << imagesFile << "\n";
    cout << "reading MNIST labels from " << labelsFile << "\n";
    LabeledDataset mnist = IDXReader().read(trainImages, trainLabels);
    return mnist;
}


class IntClassifier : public AClassifier < int, 1 > {
private:
	OneHotEncoder &enc;
	Net &net;

public:
	IntClassifier(Net &net, OneHotEncoder &enc)
		: enc(enc), net(net) {}

	virtual int aslabel(const Output &out) {
		return enc.unapply(out)[0];
	}
	virtual int classify(const Input &in) {
		return enc.unapply(net.output(in))[0];
	}
};

float meanLossEstimate(Net &net, LabeledDataset &dataset, size_t maxcount=0) {
    float total;
    size_t n = 0;
    for (auto sample : dataset) {
        total += net.loss(sample.data, sample.label);
        n++;
        if (maxcount > 0 && n >= maxcount) {
            break;
        }
    }
    return total / n;
}

void printStats(Net &net, LabeledDataset &testSet, IntClassifier &metrics) {
    auto cm = metrics.test(testSet);
    cout << "E = "
        << setprecision(6)
        << meanLossEstimate(net, testSet) << " "
        << "PPV = "
        << setprecision(3)
        << cm.precision() << " "
        << "TPR = "
        << setprecision(3)
        << cm.recall() << " "
        << "ACC = "
        << setprecision(3)
        << cm.accuracy() << " "
        << "F1 = "
        << setprecision(3)
        << cm.F1score() << endl;
}

int main() {
    LabeledDataset mnist = readMNIST(IMAGES_FILE, LABELS_FILE);
    LabeledDataset mnistTest = readMNIST(IMAGES_TEST_FILE, LABELS_TEST_FILE);
    LabeledDataset mnistMiniTest = readMNIST(IMAGES_TEST_FILE, LABELS_TEST_FILE);

    OneHotEncoder onehot(mnist.getLabels());
    mnist.applyToLabels(onehot);
    mnistTest.applyToLabels(onehot);
    mnistMiniTest.applyToLabels(onehot);

    // test on a small dataset while running
    unique_ptr<RNG> rng = Init::newRNG();
    mnistMiniTest.shuffle(rng);
    mnistMiniTest.truncate(1000);

    Net net = MakeNet(mnist.inputDim())
        .addFC(300, scaledTanh)
        .addFC(100, leakyReLU)
        .addFC(mnist.outputDim(), scaledTanh)
        .addSoftmax()
        .init();

    IntClassifier metrics(net, onehot);
    net.setLearningPolicy(AdaDelta());
    SGD::train(net, mnist, 3 /* epochs */,
               EpochCallback { 1, [&](int i) {
                   cout << "epoch " << i << ": ";
                   printStats(net, mnistTest, metrics);
                   return false;
               }},
               IterationCallback { 1000, [&](int i) {
                   cout << "sample " << i << ": ";
                   printStats(net, mnistMiniTest, metrics);
                   return false;
               }});

    ofstream nnfile(SAVE_NETWORK_SNAPSHOT);
    if (nnfile.is_open()) {
        PlainTextNetworkWriter(nnfile) << net << "\n";
        cout << "wrote " << SAVE_NETWORK_SNAPSHOT << "\n";
    }
}
