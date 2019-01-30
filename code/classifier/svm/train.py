'''
Application for training a classifier model.
Usage: test in_path labels_path model_path image_height image_width c gamma cell_height cell_width block_height block_width bin_count [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import pickle
import sys
import sklearn.svm
import sklearn.metrics
import classifier.svm.input as inp
import classifier.svm.model as model


def train(in_path, labels_path, model_path, image_size, c, gamma, cell_size, block_size, bin_count, extension = ""):
    '''
    Trains and writes to file a multi-class classifier.

    Arguments:
        in_path     -- The path to the directory containing the dataset's images.
        labels_path -- The path to the JSON file containing label information.
        model_path  -- The path to the file where to store the pickled model.
        image_size  -- The size to which to resize the images.
        c           -- The C parameter for the SVM classifier.
        gamma       -- The gamma parameter for the SVM classifier.
        cell_size   -- The cell size for HOG feature calculation.
        block_size  -- The block size for HOG feature calculation.
        bin_count   -- The number of bins for HOG feature calculation.
        extension   -- The file type extension of the images to load.
    '''

    _, images, labels = inp.load_images(in_path, labels_path, image_size, extension = extension)
    features = inp.compute_features(images, cell_size, block_size, bin_count)

    classifier = model.train(features, labels, c, gamma)

    print("Error: ", model.test(features, labels, classifier).error)

    classifier.hog_image_size = image_size
    classifier.hog_cell_size = cell_size
    classifier.hog_block_size = block_size
    classifier.hog_bin_count = bin_count

    with open(model_path, 'wb') as model_file: pickle.dump(classifier, model_file)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Model path:\t", end = '')
        argv.append(input())
        print("Image height:\t", end = '')
        argv.append(input())
        print("Image width:\t", end = '')
        argv.append(input())
        print("C:\t", end = '')
        argv.append(input())
        print("gamma:\t", end = '')
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

    if len(argv) in (12, 13):
        in_path = argv[0]
        labels_path = argv[1]
        model_path = argv[2]

        image_size = [int(argv[3]), int(argv[4])]
        c = float(argv[5])
        gamma = float(argv[6])
        cell_size = (int(argv[7]), int(argv[8]))
        block_size = (int(argv[9]), int(argv[10]))
        bin_count = int(argv[11])
        extension = argv[12] if len(argv) > 12 else ""

        train(in_path, labels_path, model_path, image_size, c, gamma, cell_size, block_size, bin_count, extension = extension)
    else:
        print("Usage: test in_path labels_path model_path image_height image_width c gamma cell_height cell_width block_height block_width bin_count [extension]")


if __name__ == '__main__':
    main(sys.argv)