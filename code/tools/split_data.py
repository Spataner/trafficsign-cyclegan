'''
Application for creating training/test dataset splits.
Usage: split_data in_path out_path_training out_path_test training_percentage [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import sys
import shutil
import random


def split_data(in_path: str, out_path_training: str, out_path_test, split: float, extension: str = ""):
    '''
    Splits a dataset of files in a directory into training and test datasets.

    Arguments:
        in_path             -- The path to the directory which contains the examples as individual files.
        out_path_training   -- The path to the directory which is supposed to hold the training examples of the split.
        out_path_test       -- The path to the directory which is supposed to hold the test examples of the split.
        split               -- A value in (0, 1) indicating the percentage of all examples used for training (with the remaining examples used for testing).
        extension           -- The file type extension of the examples.

    Returns:
        The number of training examples and the number of test examples that resulted from the split.
    '''

    if split <= 0.0 or split >= 1.0:
        raise Exception("Invalid split percentage.")

    if not os.path.isdir(in_path):
        raise Exception("Invalid input path.")

    files = [element for element in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, element)) and element.endswith(extension)]

    trainingExCount = round(len(files) * split)
    testExCount = len(files) - trainingExCount

    random.shuffle(files)

    if not os.path.isdir(out_path_training):
        os.mkdir(out_path_training)

    if not os.path.isdir(out_path_test):
        os.mkdir(out_path_test)

    for i in range(trainingExCount):
        shutil.copyfile(os.path.join(in_path, files[i]), os.path.join(out_path_training, files[i]))

    for i in range(trainingExCount, testExCount+trainingExCount):
        shutil.copyfile(os.path.join(in_path, files[i]), os.path.join(out_path_test, files[i]))

    return trainingExCount, testExCount

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t\t")
        argv.append(input())

        print("Training output path:\t")
        argv.append(input())

        print("Test output path:\t")
        argv.append(input())

        print("Split:\t\t\t")
        argv.append(input())

        print("Extension:\t\t")
        argv.append(input())

    if len(argv) in (4, 5):
        argv[3] = float(argv[3])
        trainingExCount, testExCount = split_data(*argv)
        print(argv[0] + "@" + str(trainingExCount + testExCount))
        print("\t" + argv[1] + "@" + str(trainingExCount))
        print("\t" + argv[2] + "@" + str(testExCount))
        
    else:
        print("Usage: split_data in_path out_path_training out_path_test training_percentage [extension]")


if __name__ == '__main__':
    main(sys.argv)