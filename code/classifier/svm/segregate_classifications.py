'''
Application for segregating images based on classfication results.
Usage: segregate_classifications in_path labels_path model_path out_path [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import pickle
import os
import shutil
import classifier.svm.input as inp
from classifier.common.report import LABEL_NAMES


def segregate_classifications(in_path, labels_path, model_path, out_path, extension = ""):
    '''
    Segregates images into nested folders based on actual and predicted labels.

    Arguments:
        in_path     -- The path to the directory containing the dataset's images.
        labels_path -- The path to the JSON file containing label information.
        model_path  -- The path to the file containing the pickled model.
        out_path    -- The path to the directory where to build the nested folder structure.
        extension   -- The file type extension of the images to load.
    '''

    with open(model_path, 'rb') as model_file: classifier = pickle.load(model_file)

    file_names, images, labels = inp.load_images(in_path, labels_path, classifier.hog_image_size, extension = extension)
    features = inp.compute_features(images, classifier.hog_cell_size, classifier.hog_block_size, classifier.hog_bin_count)

    predicted_labels = classifier.predict(features)

    for i in range(len(file_names)):
        if not os.path.isdir(os.path.join(out_path, LABEL_NAMES[labels[i]], LABEL_NAMES[predicted_labels[i]])):
            os.makedirs(os.path.join(out_path, LABEL_NAMES[labels[i]], LABEL_NAMES[predicted_labels[i]]))

        shutil.copyfile(os.path.join(in_path, file_names[i]), os.path.join(out_path, LABEL_NAMES[labels[i]], LABEL_NAMES[predicted_labels[i]], file_names[i]))

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Model path:\t", end = '')
        argv.append(input())
        print("Output path:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (4, 5):
        in_path = argv[0]
        labels_path = argv[1]
        model_path = argv[2]
        out_path = argv[3]

        extension = argv[4] if len(argv) > 4 else ""

        segregate_classifications(in_path, labels_path, model_path, out_path, extension = extension)
    else:
        print("Usage: segregate_classifications in_path labels_path model_path out_path [extension]")


if __name__ == '__main__':
    main(sys.argv)