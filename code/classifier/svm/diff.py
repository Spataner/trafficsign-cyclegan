'''
Application for comparing two classifier models.
Usage: test in_path labels_path model_path1 model_path2 [extension]

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


def diff(in_path, labels_path, model_path1, model_path2, report_path, extension = ""):
    '''
    Outputs the difference in error of two classifier models on a given dataset.

    Arguments:
        in_path     -- The path to the directory containing the dataset's images.
        labels_path -- The path to the JSON file containing label information.
        model_path1  -- The path to the file containing the first pickled model.
        model_path2  -- The path to the file containing the second pickled model.
        report_path -- The path to the file where to store the error report.
        extension   -- The file type extension of the images to load.
    '''

    with open(model_path1, 'rb') as model_file: classifier1 = pickle.load(model_file)
    with open(model_path2, 'rb') as model_file: classifier2 = pickle.load(model_file)

    _, images, labels = inp.load_images(in_path, labels_path, classifier1.hog_image_size, extension = extension)
    features = inp.compute_features(images, classifier1.hog_cell_size, classifier1.hog_block_size, classifier1.hog_bin_count)

    report1 = model.test(features, labels, classifier1)
    report2 = model.test(features, labels, classifier2)

    report_diff = report1 - report2

    print("Error: ", report_diff.error)
    print()
    print(report_diff.get_sparse_string(report_diff.confusion_matrix_relative, 0.01))

    with open(report_path, 'w') as report_file: report_diff.dump(report_file)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Model path 1:\t", end = '')
        argv.append(input())
        print("Model path 2:\t", end = '')
        argv.append(input())
        print("Report path:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (5, 6):
        in_path = argv[0]
        labels_path = argv[1]
        model_path1 = argv[2]
        model_path2 = argv[3]
        report_path = argv[4]


        extension = argv[5] if len(argv) > 5 else ""

        diff(in_path, labels_path, model_path1, model_path2, report_path, extension = extension)
    else:
        print("Usage: diff in_path labels_path model_path1 model_path2 [extension]")


if __name__ == '__main__':
    main(sys.argv)