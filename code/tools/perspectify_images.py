'''
Application for applying randomized perspective perturbations to a collection of iamges
Usage: perspectify_images in_path out_path target_height target_width border_percent_lower_bound
       border_percent_upper_bound  border_balance_lower_bound border_balance_upper_bound perturbation_lower_bound
       perturbation_upper_bound perspectify_count [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import random
import sys
import pickle
import numpy as np
import cv2


def in_image(row, column, shape):
    '''
    Determines whether a certain pixel position lies within the image boundaries.

    Arguments:
        row     -- The row of the position to check.
        column  -- The column of the position to check.
        shape   -- The image shape.

    Returns:
        True, if the position is within the image boundaries; False, otherwise.
    '''

    return row >= 0 and row <= shape[0]-1 and column >= 0 and column <= shape[1]-1

def borders_transparency(image, row, column):
    '''
    Determines whether the pixel at position (row, column) has a transparent pixel in its 4-neighborhood.
    Pixels outside the image boundary are considered transparent for this purpose.

    Arguments:
        image   -- The image for which to perform the pixel check.
        row     -- The row of the pixel for which to perform the check.
        column  -- The column of the pixel for which to perform the check.

    Returns:
        True, if the pixel borders a transparent pixel; False, otherwise.
    '''

    for i in (row-1, row+1):
        if not in_image(i, column, image.shape) or image[i, column, 2] == 0:
            return True

    for j in (column-1, column+1):
        if not in_image(row, j, image.shape) or image[row, j, 2] == 0:
            return True

    return False

def determine_transparency_border(image):
    '''
    Determines the list of pixel positions that are non-transparent but have a transparent pixel in their
    4-neighborhood. Pixel positions outside the image boundaries are treated as transparent for this purpose.

    Arguments:
        image   -- The image for which to determines the transparency border.

    Returns:
        A list of pixel positions as numpy arrays of the form [column, row, 1] (to go with OpenCV's definition of
        transformation matrices).
    '''

    border_pixels = list()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 2] != 0 and borders_transparency(image, i, j):
                border_pixels.append(np.array([j, i, 1]))

    return border_pixels

def get_corrected_transform(image, transform, border_pixels):
    '''
    Corrects a transform matrix such that the non-transparent contents of an image are fully within the positive
    coordinate range of the transformed image space.

    Arguments:
        image           -- The image for which to assure that its contents are in the positive quadrant.
        transform       -- The 3x3 transform matrix to correct.
        border_pixels   -- The pixels of the border of the non-transparent image contents.

    Returns:
        The corrected transform matrix as well as a tuple specifying the image size required to fully contain the
        transformed non-transparent image contents.
    '''

    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')

    for pixel in border_pixels:
        transformed_pixel = np.matmul(transform, pixel)
        transformed_pixel /= transformed_pixel[2]

        min_x = min(min_x, transformed_pixel[0])
        max_x = max(max_x, transformed_pixel[0])
        min_y = min(min_y, transformed_pixel[1])
        max_y = max(max_y, transformed_pixel[1])

    translation = np.identity(3)
    translation[0, 2] = -min_x
    translation[1, 2] = -min_y

    transform = np.matmul(translation, transform)
    dsize = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))

    return transform, dsize

def get_random_perspective_transform(perturbation_range, image_shape):
    '''
    Creates a random perspective transform by slightly perturbing the corners of the unit square and deriving from
    this the transformation matrix.

    Arguments:
        perturbation_range  -- The range for the perturbation applied to the coordinates.

    Returns:
        A 3x3 perspective transform matrix.
    '''

    source_points = np.array([[0., 0.], [0., image_shape[0]], [image_shape[1], 0.], [image_shape[1], image_shape[0]]], dtype = np.float32)
    size_average = (image_shape[0] + image_shape[1]) / 2
    target_points = source_points + np.random.uniform(low = perturbation_range[0] * size_average, high = perturbation_range[1] * size_average, size = [4, 2]).astype(np.float32)

    return cv2.getPerspectiveTransform(source_points, target_points)


class Instance:
    '''
    Class representing one input image.
    '''

    def __init__(self, in_path, file_name):
        self.file_name = file_name
        self.image = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_UNCHANGED)

        if os.path.isfile(os.path.join(in_path, file_name + "_border.pickle")):
            with open(os.path.join(in_path, file_name + "_border.pickle"), 'rb') as border_file: self.border_pixels = pickle.load(border_file)
        else:
            self.border_pixels = determine_transparency_border(self.image)
            with open(os.path.join(in_path, file_name + "_border.pickle"), 'wb') as border_file: pickle.dump(self.border_pixels, border_file)


def perspectify_images(in_path, out_path, target_size, border_percent_range, border_balance_range, perturbation_range, perspectify_count, extension = ""):
    '''
    Apply randomized perspective perturbations to a collection of images.

    Arguments:
        in_path                 -- The path to the directory containing the input images.
        out_path                -- The path to the directory to which the output images are saved.
        target_size             -- The desired output image size.
        border_percent_range    -- The range in percent of height and width pixels left empty around the image contents.
        border_balance_range    -- The range in percent how distribution of border pixels is balanced between top/bottom
                                   and left/right.
        perturbation_range      -- The percentage range for the perturbation to the corners of the image from which the
                                   random perspective transform matrix is derived.
        perspectify_count       -- The number of perturbed images created per input image.
        extension               -- The file type extension of the images to load.
    '''

    instances = list()
        
    for file_name in os.listdir(in_path):
        if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension):
            instances.append(Instance(in_path, file_name))

    for instance in instances:
        for i in range(perspectify_count):
            border_percent = random.uniform(border_percent_range[0], border_percent_range[1])
            border_balance_vertical = random.uniform(border_balance_range[0], border_balance_range[1])
            border_balance_horizontal = random.uniform(border_balance_range[0], border_balance_range[1])

            central_target_size = [int(round(target_size[0] * (1. - 2. * border_percent))), int(round(target_size[1] * (1. - 2. * border_percent)))]

            transform = get_random_perspective_transform(perturbation_range, instance.image.shape)
            transform, dsize = get_corrected_transform(instance.image, transform, instance.border_pixels)
            transformed_image = cv2.warpPerspective(instance.image, transform, dsize, flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = (0, 0, 0, 0))

            scale = min(central_target_size[0] / transformed_image.shape[0], central_target_size[1] / transformed_image.shape[1])

            central_image = cv2.resize(transformed_image, (int(round(scale * transformed_image.shape[1])), int(round(scale * transformed_image.shape[0]))), interpolation = cv2.INTER_CUBIC)

            borders = [target_size[0] - central_image.shape[0], target_size[1] - central_image.shape[1]]

            border_top = int(round(borders[0] * border_balance_vertical))
            border_bottom = borders[0] - border_top

            border_top, border_bottom = random.sample([border_top, border_bottom], 2)

            border_left = int(round(borders[1] * border_balance_horizontal))
            border_right = borders[1] - border_left

            border_left, border_right = random.sample([border_left, border_right], 2)

            target_image = cv2.copyMakeBorder(central_image, border_top, border_bottom, border_left, border_right, cv2.BORDER_CONSTANT, value = (0, 0, 0, 0))

            new_file_name = instance.file_name[:instance.file_name.rfind(extension)] + "_" + str(i) + extension
            cv2.imwrite(os.path.join(out_path, new_file_name), target_image)

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
        print("Border percent:\t", end = '')
        argv.append(input())
        print("Perturbation lower bound:\t", end = '')
        argv.append(input())
        print("Perturbation upper bound:\t", end = '')
        argv.append(input())
        print("Perspectify count:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (11, 12):
        in_path = argv[0]
        out_path = argv[1]
        target_size = (int(argv[2]), int(argv[3]))
        border_percent_range = (float(argv[4]), float(argv[5]))
        border_balance_range = (float(argv[6]), float(argv[7]))
        perturbation_range = (float(argv[8]), float(argv[9]))
        perspectify_count = int(argv[10])
        extension = argv[11] if len(argv) > 11 else ""

        perspectify_images(in_path, out_path, target_size, border_percent_range, border_balance_range, perturbation_range, perspectify_count, extension = extension)
    else:
        print(
            "Usage: perspectify_images in_path out_path target_height target_width border_percent_lower_bound\n\
             border_percent_upper_bound  border_balance_lower_bound border_balance_upper_bound perturbation_lower_bound\n\
             perturbation_upper_bound perspectify_count [extension]"
        )


if __name__ == '__main__':
    main(sys.argv)