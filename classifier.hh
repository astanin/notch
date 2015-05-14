#ifndef CLASSIFIER_H
#define CLASSIFIER_H


#include "dataset.hh"


struct ConfusionMatrix {
    int truePositives  = 0;
    int falsePositives = 0; // type I error
    int trueNegatives  = 0;
    int falseNegatives = 0; // type II error

    double recall() {
        return 1.0 * truePositives / (truePositives + falseNegatives);
    }

    double precision() {
        return 1.0 * truePositives / (truePositives + falsePositives);
    }

    double accuracy() {
        double totalTrue = (truePositives + trueNegatives);
        double totalFalse = (falsePositives + falseNegatives);
        return totalTrue / (totalTrue + totalFalse);
    }

    double F1score() {
        double p = precision();
        double r = recall();
        return 2.0 * p * r / (p + r);
    }
};


template <typename Out> class Classifier {
public:
    virtual Out classify(const Input &input) = 0;
    virtual ConfusionMatrix test(const LabeledDataset &testSet) = 0;
};


/// Binary classifier returns two class labels: true and false.
class BinaryClassifier : public Classifier<bool> {
public:
    virtual ConfusionMatrix test(const LabeledDataset &testSet) {
        assert(testSet.outputDim() == 1);
        ConfusionMatrix cm;
        for (LabeledData sample : testSet) {
            bool result = this->classify(sample.data);
            bool expected = sample.label[0] > 0;
            if (expected) {
                if (result) {
                    cm.truePositives++;
                } else {
                    cm.falseNegatives++;
                }
            } else {
                if (result) {
                    cm.falsePositives++;
                } else {
                    cm.trueNegatives++;
                }
            }
        }
        return cm;
    }
};


class LinearPerceptronClassifier : public BinaryClassifier {
private:
    LinearPerceptron &perceptron;

public:
    LinearPerceptronClassifier(LinearPerceptron &perceptron)
        : perceptron(perceptron) {}

    virtual bool classify(const Input &x) {
        return perceptron.output(x) > 0;
    }
};

#endif
