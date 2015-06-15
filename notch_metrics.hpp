#ifndef NOTCH_METRICS_H
#define NOTCH_METRICS_H

/** @file notch_metrics.hpp Optional performance metrics.
 *
 * This file contains code to calculate confusion matrix of a classifier
 * and estimate the classifier's accuracy, precision and F-score.
 *
 * References:
 *
 * - A systematic analysis of performance measures for classification tasks
 *   (2009) Sokolova and Lapalme
 * - http://stats.stackexchange.com/a/51301
 **/

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

#include <map>
#include <utility>  // pair


#include "notch.hpp"


/** A multi-class confusion matrix for class labels of type C. */
template<typename C, C default_class = C()>
class ConfusionMatrix {
	/// Confusion matrix cm is a map from (actual_class, calculated_class)
	/// to the number of observations; we may think about it as a sparse
	/// matrix where rows are actual_class, and columns are calculated_class.
	std::map<std::pair<C, C>, int> cm;

public:

	void add(C actual, C predicted) {
		std::pair<C, C> key = std::make_pair(actual, predicted);
		cm[key]++;
	}

	/// in binary case, recall = tp / (tp + fn);
	double recall(C class_ = default_class) {
		int class_total_samples = 0;
		int class_positive_results = 0;
		for (auto p : cm) {
			auto actual = p.first.first;
			auto predicted = p.first.second;
			auto count = p.second;
			if (actual == class_) {
				class_total_samples += count;
				if (predicted == actual) {
					class_positive_results += count;
				}
			}
		}
		return 1.0 * class_positive_results / class_total_samples;
    }

	/// in binary case, precision = tp / (tp + fp);
	double precision(C class_ = default_class) {
		int hypothesis_total_results = 0;
		int hypothesis_true_results = 0;
		for (auto p : cm) {
			auto actual = p.first.first;
			auto predicted = p.first.second;
			auto count = p.second;
			if (predicted == class_) {
				hypothesis_total_results += count;
				if (predicted == actual) {
					hypothesis_true_results += count;
				}
			}
		}
		return 1.0 * hypothesis_true_results / hypothesis_total_results;
    }

	/// in binary case, accuracy = (tp + tn) / (fp + fn);
    double accuracy() {
		int total_true = 0;
		int total = 0;
		for (auto p : cm) {
			auto actual = p.first.first;
			auto predicted = p.first.second;
			auto count = p.second;
			total += count;
			if (actual == predicted) {
				total_true += count;
			}
		}
        return 1.0 * total_true / total;
    }

	/// in binary case, F1 = 2 * precision * recall / (precision + recall);
    double F1score(C class_ = default_class) {
        double p = precision(class_);
        double r = recall(class_);
        return 2.0 * p * r / (p + r);
    }
};

/** A classifier assigns class labels of type C to input vetors. */
template<class C, C default_class = C()>
class AClassifier {
public:
	virtual C aslabel(const Output &output) = 0;
	virtual C classify(const Input &input) = 0;
	virtual ConfusionMatrix<C, default_class> test(const LabeledDataset &testSet) {
		ConfusionMatrix<C, default_class> cm;
		for (LabeledData sample : testSet) {
			C predicted = classify(sample.data);
			C actual = aslabel(sample.label);
			cm.add(actual, predicted);
		}
		return cm;
	}
};

#endif /* NOTCH_METRICS_H */
