#!/usr/bin/env python
"""Usage: python images2idx.py [options] PREFIX CLASS_0_DIR CLASS_1_DIR ...

Create a dataset in IDX format (same as MNIST dataset) from images saved to
different directories, one directory per class (images of the first class
should be saved in CLASS_0_DIR and will be assigned label 0, images of the
second class should be saved to CLASS_1_DIR and will be assigned label 1, and
so on).  The script will create files named 'PREFIX-images-idx3-ubyte' and
'PREFIX-labels-idx1-ubyte'.

For more information about IDX format see http://yann.lecun.com/exdb/mnist/

Options:

--mode MODE   color mode, 'rgb' or 'gray' (default: gray)
-W WIDTH
-H HEIGHT     resize all image samples to WIDTHxHEIGHT pixels
              (default: use the size of the first image)

"""


from __future__ import print_function
import sys
import getopt
import os

import numpy as np
from PIL import Image


def list_of_images(image_dirs):
    "Return a list of (class_id, path_to_image) pairs."
    class_id = 0
    image_list = []
    for d in image_dirs:
        for f in os.listdir(d):
           if f.lower().endswith(".png") \
              or f.lower().endswith(".jpg") \
              or f.lower().endswith(".jpeg"):
                full_path = os.path.join(d, f)
                image_list.append((class_id, full_path))
        class_id += 1
    return image_list


def write_images_header(imagesfile, n, width, height, mode):
    channels = len(mode)  # 3 for RGB, 1 for L, 4 for CMYK and RGBA
    num_dimensions = 4 if channels > 1 else 3
    magic = [0, 0]
    magic.append(0x08)   # unsigned byte data
    magic.append(num_dimensions)
    # write magic:
    imagesfile.write("".join([chr(0xFF & c) for c in magic]))
    # write dimensions
    if channels > 1:
        dimensions = [n, channels, width, height]
    else:
        dimensions = [n, width, height]
    for d in dimensions:
        # write each dimension in MSB first, high endian order
        imagesfile.write("".join(
            [chr((d & 0xFF000000) >> 24),
             chr((d &   0xFF0000) >> 16),
             chr((d &     0xFF00) >> 8),
             chr((d &       0xFF))]))


def write_labels_header(labelsfile, n):
    magic = [0, 0, 0x08, 0x01]
    labelsfile.write("".join([chr(0xFF & c) for c in magic]))
    labelsfile.write("".join(
            [chr((n & 0xFF000000) >> 24),
             chr((n &   0xFF0000) >> 16),
             chr((n &     0xFF00) >> 8),
             chr((n &       0xFF))]))


def convert_images(image_dirs, prefix, width, height, mode):
    imagesfilename = prefix + "-images-idx3-ubyte"
    labelsfilename = prefix + "-labels-idx1-ubyte"
    if not width or not height:
        raise NotImplementedError("No automatic resizing. Use -W and -H options")
    images = list_of_images(image_dirs)
    n = len(images)
    with open(imagesfilename, "wb") as imagesfile:
        with open(labelsfilename, "wb") as labelsfile:
            write_images_header(imagesfile, n, width, height, mode)
            write_labels_header(labelsfile, n)
            for class_id, image_path in images:
                img = Image.open(image_path)
                img = img.convert(mode=mode)
                imagesfile.write(np.asarray(img.getdata()))
                imagesfile.flush()
                labelsfile.write(chr(class_id))
                labelsfile.flush()


def main():
    args = sys.argv[1:]
    opts, args = getopt.getopt(args, 'hH:W:', ['help', 'mode='])
    opts = dict(opts)
    n_classes = len(args) - 1
    if n_classes < 2 or "-h" in opts or "--help" in opts:
        print(__doc__)
        exit(0)
    width = int(opts["-W"]) if "-W" in opts else 0
    height = int(opts["-H"]) if "-H" in opts else 0
    mode = opts["--mode"] if "--mode" in opts else "gray"
    if mode == "gray":
        mode = "L"  # PIL/Pillow mode name for grayscale images
    prefix = args[0]
    dirs = args[1:]
    convert_images(dirs, prefix, width, height, mode)


if __name__ == "__main__":
    main()
