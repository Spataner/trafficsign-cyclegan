'''
Application for resampling a set of files such that all classes are balanced.
Usage: resample in_path labels_path out_path [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import sys
import shutil
import json


def resample(in_path, labels_path, out_path, extension = ""):
    '''
    Resample a collection of files such that all classes are balanced.

    Arguments:
        in_path     -- The path to the directory from which to take the input files.
        labels_path -- The path tot he JSON file containing the class labels for the files.
        out_path    -- The path to the directory where to output the resampled collection of images.
        extension   -- The file type extension of the files to resample.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    with open(labels_path, 'r') as label_file: labels = json.load(label_file)

    inverse_labels = dict()

    for label in set(labels.values()):
        inverse_labels[label] = list()

    for file_name in file_names:
        inverse_labels[labels[file_name]].append(file_name)

    target_count = max([len(inverse_labels[label]) for label in inverse_labels.keys()])

    new_labels = dict()

    for label in inverse_labels.keys():
        for i in range(target_count):
            old_file_name = inverse_labels[label][i % len(inverse_labels[label])]
            new_file_name = old_file_name[:old_file_name.rfind(extension)] + "_" + str(i // len(inverse_labels[label])) + extension
            shutil.copyfile(os.path.join(in_path, old_file_name), os.path.join(out_path, new_file_name))
            new_labels[new_file_name] = labels[old_file_name]

    with open(os.path.join(out_path, "labels.json"), 'w') as labels_file: json.dump(new_labels, labels_file)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Labels path:\t", end = '')
        argv.append(input())
        print("Output path:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (3, 4):
        resample(*argv)
    else:
        print("Usage: resample in_path labels_path out_path [extension]")


if __name__ == '__main__':
    main(sys.argv)