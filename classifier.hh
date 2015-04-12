#ifndef CLASSIFIER_H
#define CLASSIFIER_H


#include "dataset.hh"


struct ConfusionMatrix {
    int truePositives;
    int falsePositives; // type I error
    int trueNegatives;
    int falseNegatives; // type II error

    ConfusionMatrix() :
        truePositives(0), falsePositives(0),
        trueNegatives(0), falseNegatives(0) {}

    double recall() {
        return 1.0*truePositives/(truePositives + falseNegatives);
    }

    double precision() {
        return 1.0*truePositives/(truePositives + falsePositives);
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


template<typename Out>
class Classifier {
    public:
        virtual Out classify(const Input& input) const = 0;
        virtual ConfusionMatrix test(const LabeledSet& testSet) const = 0;
};


/// Binary classifier returns two class labels: true and false.
class BinaryClassifier : public Classifier<bool> {
    public:
        virtual ConfusionMatrix test(const LabeledSet& testSet) const {
            assert (testSet.getOutputSize() == 1);
            ConfusionMatrix cm;
            for (LabeledPair sample : testSet) {
                bool result = this->classify(sample.input);
                bool expected = sample.output[0] > 0;
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


#endif
