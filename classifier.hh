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
        virtual Out response(const Input& input) const = 0;
        virtual ConfusionMatrix test(const LabeledSet& testSet) const = 0;
};


template<typename ScalarType>
class BinaryClassifier : public Classifier<ScalarType> {
    public:
        virtual ConfusionMatrix test(const LabeledSet& testSet) const {
            assert (testSet.getOutputSize() == 1);
            ConfusionMatrix cm;
            for (LabeledPair sample : testSet) {
                auto result = this->response(sample.input);
                auto expected = sample.output[0];
                bool epectedIsPositive = expected > 0;
                bool resultIsPositive = result > 0;
                if (epectedIsPositive) {
                    if (resultIsPositive) {
                        cm.truePositives++;
                    } else {
                        cm.falseNegatives++;
                    }
                } else {
                    if (resultIsPositive) {
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
