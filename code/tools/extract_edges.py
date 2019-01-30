'''
Application for extracting edges from a set of images (using Canny edge extractor).
Requires OpenCV.
Usage: extract_edges in_path out_path

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import os
import cv2


BINARY_SLACK = 5    #How far (in terms of uint8 gray-values) the colors of the binary edge images are from true white/black.


def extract_edges(in_path, out_path, extension = ""):
    '''
    Extracts the edges from a set of images.

    Arguments:
        in_path     -- The path to the directory from which to load the images.
        out_path    -- The path to the directory where to save the edge images.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name))
        edges = cv2.Canny(image, 100, 200)
        inverted = cv2.bitwise_not(edges)
        _, truncated = cv2.threshold(inverted, 255 - 2*BINARY_SLACK, 255, cv2.THRESH_TRUNC)
        truncated_color = cv2.cvtColor(truncated + BINARY_SLACK, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(os.path.join(out_path, file_name), truncated_color)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Output path:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (2, 3):
        extract_edges(*argv)
    else:
        print("Usage: extract_edges in_path out_path [extension]")


if __name__ == '__main__':
    main(sys.argv)