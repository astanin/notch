#!/usr/bin/env python
"""Usage: python fann2csv.py datafile.fann

Convert FANN dataset file 'datafile.fann' to a CSV file 'datafile.csv'.
Class labels are saved in the last columns.

For more information about FANN format see
http://leenissen.dk/fann/html/files2/gettingstarted-txt.html

"""


from __future__ import print_function
import sys
import numpy as np


def convert_fann_to_csv(fannfilename):
    csvfilename = fannfilename.rsplit(".")[0] + ".csv"
    with open(fannfilename) as fannfile:
        lines = fannfile.read().splitlines()
        datalines = lines[1::2]
        labellines = lines[2::2]
        data = np.asarray([map(float, l.split()) for l in datalines])
        labels = np.asarray([map(float, l.split()) for l in labellines])
        table = np.hstack((data, labels))
        np.savetxt(csvfilename, table, delimiter=",", fmt="%g")


def main():
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        exit(0)
    for f in args:
        convert_fann_to_csv(f)


if __name__ == "__main__":
    main()
