#!/usr/bin/env python
"""Download and unpack MNIST dataset."""

from __future__ import print_function
import urllib2
import gzip
import os


IMAGES_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
LABELS_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMAGES_URL = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS_URL = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"


def download(url, filename):
    u = urllib2.urlopen(url)
    with open(filename + ".gz", "wb") as gf:
        gf.write(u.read())
        print("downloaded " + filename + ".gz")
    with gzip.GzipFile(filename + ".gz") as gf:
        with open(filename, "wb") as f:
            f.write(gf.read())
            print("saved " + filename)
    os.unlink(filename + ".gz")


def main():
    download(IMAGES_URL, "train-images-idx3-ubyte")
    download(LABELS_URL, "train-labels-idx1-ubyte")
    download(TEST_IMAGES_URL, "t10k-images-idx3-ubyte")
    download(TEST_LABELS_URL, "t10k-labels-idx1-ubyte")


if __name__ == "__main__":
    main()
