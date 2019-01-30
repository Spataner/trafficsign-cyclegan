'''
Classifier input and HOG feature operations.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import json
import numpy as np
import cv2



def load_images(in_path, labels_path, image_size, extension = ""):
    '''
    Loads, resizes, and converts to grayscale a collection of images alongside label information.

    Arguments:
        in_path     -- The path to the input directory for the image.
        labels_path -- The path to the JSON file containing the label information.
        image_size  -- The size to which the images are resized.
        extension   -- The file type extension for the image to load.

    Returns:
        Two numpy arrays, one containing the images, the other the labels.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    images = np.zeros([len(file_names)] + image_size)

    with open(labels_path, 'r') as labels_file: labels_dict = json.load(labels_file)

    labels = np.array([labels_dict[file_name] for file_name in file_names])

    for i in range(len(file_names)):
        image = cv2.imread(os.path.join(in_path, file_names[i]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, tuple(image_size), interpolation = cv2.INTER_CUBIC)
        images[i] = image

    return file_names, images.astype(np.uint8), labels

def compute_features(images, cell_size, block_size, bin_count):
    '''
    Computes the HOG features for a set of images.

    Arguments:
        images      -- The images for which to calculate the features.
        cell_size   -- The HOG feature cell size.
        block_size  -- The HOG feature block size.
        bin_count   -- The number of HOG feature bins.

    Returns:
        A numpy array containing the HOG features.
    '''

    hog = cv2.HOGDescriptor(
        _winSize=(images.shape[2] // cell_size[1] * cell_size[1], images.shape[1] // cell_size[0] * cell_size[0]),
        _blockSize = (block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
        _blockStride = cell_size, _cellSize = cell_size, _nbins = bin_count
    )

    hog_features = list()

    for i in range(images.shape[0]):
        hog_features.append(hog.compute(images[i]))

    return np.squeeze(hog_features)

