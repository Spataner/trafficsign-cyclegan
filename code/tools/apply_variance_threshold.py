'''
Application for discarding low variance images.
Usage: apply_variance_threshold in_path out_path threshold [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import os
import numpy as np
import cv2


def apply_variance_threshold(in_path, out_path, threshold, extension = ""):
    '''
    Selects from a collection of images those that have a variance above the given threshold.

    Arguments:
        in_path         -- The path to the directory of input images.
        out_path        -- The path to the directory for the output images.
        threshold       -- The variance threshold for the images.
        extension       -- The file type extension of the images to load.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name))
        variance = np.var(image)

        if variance > threshold:
            cv2.imwrite(os.path.join(out_path, file_name), image)


def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())

        print("Output path:\t", end = '')
        argv.append(input())

        print("Threshold:\t", end = '')
        argv.append(input())

        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (3, 4):
        apply_variance_threshold(sys.argv[0], sys.argv[1], float(sys.argv[2]), extension = sys.argv[3] if len(argv) > 3 else "")
    else:
        print("Usage: apply_variance_threshold in_path out_path threshold [extension]")


if __name__ == '__main__':
    main(sys.argv)