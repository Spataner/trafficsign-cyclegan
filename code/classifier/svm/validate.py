'''
Application for validating classifier parameters.
Usage: in_path labels_path fold_count c_exponent_lower_bound c_exponent_upper_bound gamma_exponent_lower_bound gamma_exponent_upper_bound image_height image_width cell_height cell_width block_height block_width bin_count [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import numpy as np
import sklearn.utils
import classifier.svm.input as inp
import classifier.svm.model as model


def validate_once(in_features, in_labels, val_features, val_labels, c, gamma):
    '''
    Peforms one round of cross validation for a given set of parameters.

    Arguments:
        features    -- The training features.
        labels      -- The training labels.
        fold_count  -- The number of folds for cross validation.
        c           -- Parameter C for the SVM.
        gamma       -- Parameter gamma for the SVM.

    Returns:
        The cross validation error.
    '''

    classifier = model.train(in_features, in_labels, c, gamma)
    report = model.test(val_features, val_labels, classifier)

    print("C=", c, "gamma=", gamma, "error=", report.error)

    return report.error

def validate(in_path, in_labels_path, val_path, val_labels_path, c_exponent_range, gamma_exponent_range, image_size, cell_size, block_size, bin_count, extension = ""):
    '''
    Performs cross validation for ranges of the classifier parameters.

    Arguments:
        in_path                 -- The path to the input images.
        labels_path             -- The path to the class labels.
        fold_count              -- The number of folds for the cross validation.
        c_exponent_range        -- The range of values for n where parameter C=10^n.
        gamma_exponent_range    -- The range of values for n where parameter gamma=10^n.
        image_size              -- The size to which to resize the images.
        cell_size               -- The cell size for HOG feature calculation.
        block_size              -- The block size for HOG feature calculation.
        bin_count               -- The number of bins for HOG feature calculation.
        extension               -- The file type extension of the images to load.
    '''

    _, in_images, in_labels = inp.load_images(in_path, in_labels_path, image_size, extension = extension)
    in_features = inp.compute_features(in_images, cell_size, block_size, bin_count)

    _, val_images, val_labels = inp.load_images(val_path, val_labels_path, image_size, extension = extension)
    val_features = inp.compute_features(val_images, cell_size, block_size, bin_count)

    best_parameters = (None, None)
    best_error = float('inf')

    for c_exponent in range(c_exponent_range[0], c_exponent_range[1]+1):
        for gamma_exponent in range(gamma_exponent_range[0], gamma_exponent_range[1]+1):
            c = 10**c_exponent
            gamma = 10**gamma_exponent

            error = validate_once(in_features, in_labels, val_features, val_labels, c, gamma)

            if error < best_error:
                best_error = error
                best_parameters = (c, gamma)

    print(best_parameters, best_error)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Fold count:\t", end = '')
        argv.append(input())
        print("C exponent lower bound:\t", end = '')
        argv.append(input())
        print("C exponent upper bound:\t", end = '')
        argv.append(input())
        print("gamma exponent lower bound:\t", end = '')
        argv.append(input())
        print("gamma exponent upper bound:\t", end = '')
        argv.append(input())
        print("Image height:\t", end = '')
        argv.append(input())
        print("Image width:\t", end = '')
        argv.append(input())
        print("Cell height:\t", end = '')
        argv.append(input())
        print("Cell width:\t", end = '')
        argv.append(input())
        print("Block height:\t", end = '')
        argv.append(input())
        print("Block width:\t", end = '')
        argv.append(input())
        print("Bin count:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (15, 16):
        in_path = argv[0]
        in_labels_path = argv[1]
        val_path = argv[2]
        val_labels_path = argv[3]
        c_exponent_range = (int(argv[4]), int(argv[5]))
        gamma_exponent_range = (int(argv[6]), int(argv[7]))
        image_size = [int(argv[8]), int(argv[9])]
        cell_size = (int(argv[10]), int(argv[11]))
        block_size = (int(argv[12]), int(argv[13]))
        bin_count = int(argv[14])
        extension = argv[15] if len(argv) > 15 else ""

        validate(in_path, in_labels_path, val_path, val_labels_path, c_exponent_range, gamma_exponent_range, image_size, cell_size, block_size, bin_count, extension = extension)
    else:
        print("Usage: in_path labels_path fold_count c_exponent_lower_bound c_exponent_upper_bound gamma_exponent_lower_bound gamma_exponent_upper_bound image_height image_width cell_height cell_width block_height block_width bin_count [extension]")


if __name__ == '__main__':
    main(sys.argv)