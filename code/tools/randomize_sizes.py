'''
Application for randomizing the sizes of collection of images.
Usage: randomize_sizes in_path out_path min_size max_size max_ratio [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import cv2
import random
import os
import sys
import numpy as np


def randomize_size(image, min_size, max_size, max_ratio):
    '''
    Randomizes the size of an image. Preserves aspect ratio.

    Arguments:
        image       -- The image whose size to randomize.
        min_size    -- The minimal possible size.
        max_size    -- The maximal possible size.
        max_ratio   -- The maximally possible aspect ratio perturbation.

    Returns:
        The resized image.
    '''

    size = random.choice(range(min_size, max_size+1))
    size = (size, round(image.shape[0]/image.shape[1]*size))

    ratio = random.uniform(2. - max_ratio, max_ratio)
    if ratio > 1:
        ratio = 1/ratio

    resized = cv2.resize(image, size, interpolation = cv2.INTER_CUBIC)

    if random.choice([True, False]):
        new_height = int(round(resized.shape[0] * ratio))
        coord = (resized.shape[0] - new_height) // 2
        result = resized[coord:coord+new_height, :]
    else:
        new_width = int(round(resized.shape[1] * ratio))
        coord = (resized.shape[1] - new_width) // 2
        result = resized[:, coord:coord+new_width]

    return result

def randomize_sizes(in_path, out_path, min_size, max_size, max_ratio, extension = ""):
    '''
    Randomizes the sizes of a collection of images.

    Arguments:
        in_path     -- The path to the input directory for the images.
        out_path    -- The path to the output directory for the resized images.
        min_size    -- The minimal possible size.
        max_size    -- The maximal possible size.
        max_ratio   -- The maximally possible aspect ratio perturbation.
        extension   -- The file type extension of the images to load.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name))
        image = randomize_size(image, min_size, max_size, max_ratio)
        cv2.imwrite(os.path.join(out_path, file_name), image)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())

        print("Output path:\t", end = '')
        argv.append(input())

        print("Minimum size:\t\t", end = '')
        argv.append(input())

        print("Maximum size:\t\t", end = '')
        argv.append(input())

        print("Maximum ratio:\t\t", end = '')
        argv.append(input())

        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (5, 6):
        in_path = argv[0]
        out_path = argv[1]

        min_size = int(argv[2])
        max_size = int(argv[3])

        max_ratio = float(argv[4])

        extension = argv[5] if len(argv) > 5 else ""

        randomize_sizes(in_path, out_path, min_size, max_size, max_ratio, extension = extension)
    else:
        print("Usage: randomize_sizes in_path out_path min_size max_size max_ratio [extension]")


if __name__ == '__main__':
    main(sys.argv)