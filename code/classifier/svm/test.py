'''
Application for testing a classifier model.
Usage: test in_path labels_path model_path report_path [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import pickle
import sklearn.metrics
import sklearn.svm
import classifier.svm.input as inp
import classifier.svm.model as model


def test(in_path, labels_path, model_path, report_path, extension = ""):
    '''
    Outputs the error of a classifier model on a given dataset.

    Arguments:
        in_path     -- The path to the directory containing the dataset's images.
        labels_path -- The path to the JSON file containing label information.
        model_path  -- The path to the file containing the pickled model.
        report_path -- The path to the file where to store the error report.
        extension   -- The file type extension of the images to load.
    '''

    with open(model_path, 'rb') as model_file: classifier = pickle.load(model_file)

    _, images, labels = inp.load_images(in_path, labels_path, classifier.hog_image_size, extension = extension)
    features = inp.compute_features(images, classifier.hog_cell_size, classifier.hog_block_size, classifier.hog_bin_count)

    report = model.test(features, labels, classifier)

    print("Error: ", report.error)
    print()
    print(report.get_sparse_string(report.confusion_matrix_relative, 0.01))

    with open(report_path, 'w') as report_file: report.dump(report_file)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Model path:\t", end = '')
        argv.append(input())
        print("Report path:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (4, 5):
        in_path = argv[0]
        labels_path = argv[1]
        model_path = argv[2]
        report_path = argv[3]

        extension = argv[4] if len(argv) > 4 else ""

        test(in_path, labels_path, model_path, report_path, extension = extension)
    else:
        print("Usage: test in_path labels_path model_path report_path [extension]")


if __name__ == '__main__':
    main(sys.argv)