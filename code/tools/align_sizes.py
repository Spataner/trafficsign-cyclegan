'''
Application for aligning image sizes within a dataset.
Input images will be scaled as close as possible to the target size while preserving aspect ratio and then centrally cropped.
Usage: align_sizes in_path out_path target_height target_width minimal_height minimal_width [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import sys
import cv2


def discard_image(image_size, minimal_size):
    '''
    Test whether an image falls below the minimal size in at least one dimension and must be discarded.

    Arguments:
        image_size      -- A list/tuple of two integers specifying height and width of the image.
        minimal_size    -- A list/tuple of two integers specifying minimal allowable height and width.

    Returns:
        A boolean indicating whether the image shall be discarded.
    '''

    return image_size[0] < minimal_size[0] or image_size[1] < minimal_size[1]

def resize_image(image, target_size):
    '''
    Resizes an image to a target size by scaling it first while preserving aspect ratio and then centrally cropping.

    Arguments:
        image       -- The image to resize of shape [image_height, image_width, channels].
        target_size -- A list/tuple/tensor of two integers specifying target height and width.
    '''

    scale_factor = max(target_size[0] / image.shape[0], target_size[1] / image.shape[1])
    scale_size = [int(round(image.shape[0] * scale_factor)), int(round(image.shape[1] * scale_factor))]

    scaled_image = cv2.resize(image, tuple(scale_size[::-1]))
    
    crop_coord = [(scaled_image.shape[0] - target_size[0]) // 2, (scaled_image.shape[1] - target_size[1]) // 2]

    return scaled_image[crop_coord[0]:crop_coord[0] + target_size[0], crop_coord[1]:crop_coord[1] + target_size[1]]

def align_sizes(in_path, out_path, target_size, minimal_size, extension = ""):
    '''
    Align the sizes of all images within a given directory.

    Arguments:
        in_path         -- The path to the directory from which to load the images.
        out_path        -- The path to the directory where to save the resized images.
        target_size     -- A list of two integers specifying target height and width.
        minimal_size    -- A list of two integers specifying minimal allowable height and width that an image has to meet or be discarded.
        extension       -- The file type extension of the images to load.
    '''

    if not os.path.isdir(in_path):
        raise Exception("Invalid input path")

    if not os.path.isdir(out_path):
        raise Exception("Invalid output path")

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]


    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name))

        if not discard_image(image.shape, minimal_size):
            resized_image =  resize_image(image, target_size)
            cv2.imwrite(os.path.join(out_path, file_name), resized_image)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Output path:\t", end = '')
        argv.append(input())
        print("Target height:\t", end = '')
        argv.append(input())
        print("Target width:\t", end = '')
        argv.append(input())
        print("Minimal height:\t", end = '')
        argv.append(input())
        print("Minimal width:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (6, 7):
        target_size = list()
        minimal_size = list()

        target_size.append(int(argv[2]))
        target_size.append(int(argv[3]))

        minimal_size.append(int(argv[4]))
        minimal_size.append(int(argv[5]))

        argv[2:6] = [target_size, minimal_size]

        align_sizes(*argv)
    else:
        print("Usage: align_sizes in_path out_path target_height target_width minimal_height minimal_width [extension]")


if __name__ == '__main__':
    main(sys.argv)