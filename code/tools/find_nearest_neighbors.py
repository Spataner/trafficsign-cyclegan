'''
Application for finding nearest neighbors for images among a collection of reference images.
Usage: find_nearest_neighbors in_path reference_path out_path info_path [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import sys
import json
import cv2
import numpy as np


def find_nearest_neighbor(image, reference_path, reference_file_names):
    '''
    Finds the nearest neigbor in terms of Euclidean distance of an image within a collection of reference images.

    Arguments:
        image                -- The image for which to find the nearest neighbor.
        reference_path       -- The path to the directory containing the reference images.
        reference_file_names -- A list of file name heads for the reference images to load.

    Returns:
        The nearest neighbor reference image, the reference image's file name head, as well as the Euclidean distance
        between the given image and its nearest neighbor.
    '''

    nearest_reference = None
    nearest_reference_name = None
    smallest_distance = float('inf')

    for file_name in reference_file_names:
        reference = cv2.imread(os.path.join(reference_path, file_name))
        distance = np.linalg.norm(reference - image)

        if distance < smallest_distance:
            nearest_reference = reference
            nearest_reference_name = file_name
            smallest_distance = distance

    return nearest_reference, nearest_reference_name, smallest_distance

def find_nearest_neighbors(in_path, reference_path, out_path, info_path, extension = ""):
    '''
    Finds the nearest neighbors for a collection of images among a collection of reference images and outputs a collection
    of collages comparing them. Furthermore outputs a JSON file containing a mapping from input file names to nearest
    neighbor reference file names and the Euclidean distance to those nearest neighbors.

    Arguments:
        in_path         -- The path to the directory containing the input images.
        reference_path  -- The path to the directory cintaining the reference images.
        out_path        -- The path to the directory where to store the collage images.
        info_path       -- The path to the file where to store the file name mapping.
        extension       -- The file type extension of the images to load.
    '''

    input_file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]
    reference_file_names = [file_name for file_name in os.listdir(reference_path) if os.path.isfile(os.path.join(reference_path, file_name)) and file_name.endswith(extension)]

    info = dict()

    for file_name in input_file_names:
        image = cv2.imread(os.path.join(in_path, file_name))

        nearest_reference, nearest_reference_name, smallest_distance = find_nearest_neighbor(image, reference_path, reference_file_names)

        info[file_name] = [nearest_reference_name, smallest_distance]

        collage = np.concatenate((image, nearest_reference), axis = 1)
        cv2.imwrite(os.path.join(out_path, file_name), collage)

    with open(info_path, 'w') as info_file: json.dump(info, info_file)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())

        print("Reference path:\t", end = '')
        argv.append(input())

        print("Output path:\t", end = '')
        argv.append(input())

        print("Info path:\t", end = '')
        argv.append(input())

        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (4, 5):
        find_nearest_neighbors(*argv)
    else:
        print("Usage: find_nearest_neighbors in_path reference_path out_path info_path [extension]")


if __name__ == '__main__':
    main(sys.argv)


