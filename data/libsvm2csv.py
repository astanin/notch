#!/usr/bin/env python
"""Usage: python [options] libsvm2csv.py datafile

Convert dataset file 'datafile' in LIBSVM format to a CSV file 'datafile.csv'.
Class label is saved to the last column.

For more information about LIBSVM format see:
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/

Options:

-n N        number of features (default: as in the first sample)
"""


from __future__ import print_function
import sys
import numpy as np


def parse_libsvm_line(line, numfeatures='auto'):
    """Return (label, dense_array_of_features).

    >>> parse_libsvm_line("1 2:3.14 1:2.7")
    (1.0, array([ 2.7 ,  3.14]))

    >>> parse_libsvm_line("-1 3:-45", 3)
    (-1.0, array([  0.,   0., -45.]))

    """
    words = line.split()
    label = float(words[0])
    if numfeatures == 'auto':
        featuresDict = {}
        for w in words[1:]:
            if w:
                fnum, fval = w.split(':', 1)
                featuresDict[int(fnum) - 1] = float(fval)
        numfeatures = max(featuresDict.keys()) + 1
        features = np.zeros(numfeatures)
        for f in featuresDict:
            features[f] = featuresDict[f]
        return label, features
    else:
        features = np.zeros(numfeatures)
        for w in words[1:]:
            if w:
                fnum, fval = w.split(':', 1)
                idx = int(fnum) - 1
                if idx < numfeatures:
                    features[idx] = float(fval)
                else:
                    print("Warning: feature #{} ignored, use -n {} option"
                            .format(fnum, fnum),
                          file=sys.stderr)
        return label, features


def convert_libsvm_to_csv(libsvmfilename, numfeatures='auto'):
    csvfilename = libsvmfilename.rsplit(".")[0] + ".csv"
    with open(libsvmfilename) as fannfile:
        lines = fannfile.read().splitlines()
        all_labels = []
        all_features = []
        for ln in lines:
            label, features = parse_libsvm_line(ln, numfeatures)
            numfeatures = features.shape[0]
            all_labels.append(label)
            all_features.append(features)
        data = np.vstack(all_features)
        labels = np.vstack(all_labels)
        table = np.hstack((data, labels))
        np.savetxt(csvfilename, table, delimiter=",", fmt="%g")


def main():
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        exit(0)
    numfeatures = 'auto'
    if len(args) >= 2 and args[0] == "-n":
        numfeatures = int(args[1])
        args = args[2:]
    for f in args:
        convert_libsvm_to_csv(f, numfeatures)


if __name__ == "__main__":
    main()
