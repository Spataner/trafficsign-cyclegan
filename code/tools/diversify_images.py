'''
Application for diversifying an image dataset via random rotation, scaling, and recropping.
Usage: diversify_images in_path out_path target_height target_width max_angle max_ratio min_scale max_scale pad_height pad_width buffer_height buffer_width diversify_count [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import os
import random
import cv2


def random_rotate(image, max_angle):
    '''
    Applies a random rotation to the given image.

    Arguments:
        image       -- The image to rotate.
        max_angle   -- The absolute maximum rotation angle possible.

    Returns:
        The rotated image.
    '''

    angle = random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D(tuple([image.shape[i] // 2 for i in range(2)]), angle, 1.0)
    dsize = [int(image.shape[0] * abs(matrix[0, 1]) + image.shape[1] * abs(matrix[0, 0])), int(image.shape[0] * abs(matrix[0, 0]) + image.shape[1] * abs(matrix[0, 1]))]

    rotated_image = cv2.warpAffine(image, matrix, tuple(dsize), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = (255, 255, 255, 0))

    return rotated_image

def random_ratio(image, max_ratio):
    '''
    Randomly distorts the aspect ratio of the given image.

    Arguments:
        image       -- The image to distort.
        max_ratio   -- The maximum ratio of distortion that can be applied.

    Returns:
        The distorted image.
    '''

    ratio = random.uniform(1.0, max_ratio)
    scale = [1.0, ratio]
    random.shuffle(scale)
    size = [int(image.shape[i] * scale[i]) for i in range(2)]

    resized_image = cv2.resize(image, tuple(size)[::-1], interpolation = cv2.INTER_CUBIC)

    return resized_image

def random_scale(image, target_size, scale_range):
    '''
    Applies a random scale to the given image relative to a target size (aspect ratio is always preserved).

    Arguments:
        image       -- The image to scale.
        target_size -- The target size for the image.
        scale_range -- The range of possible scales relative to the target size.

    Returns:
        The scaled image.
    '''

    scale = random.uniform(*scale_range)
    shape = image.shape
    ratio = min([target_size[i] / image.shape[i] for i in range(2)])
    size = [int(shape[i] * scale * ratio) for i in range(2)]

    scaled_image = cv2.resize(image, tuple(size)[::-1], interpolation = cv2.INTER_CUBIC)

    return scaled_image

def random_recrop(image, target_size, pad_size):
    '''
    Pads and randomly recrops the given image to the target size.

    Arguments:
        image           -- The image to recrop.
        target_size     -- The size of the result image.
        pad_size        -- The size the image is padded to before cropping.

    Returns:
        The recropped image.
    '''

    pad_top = (pad_size[0] - image.shape[0]) // 2
    pad_bottom = (pad_size[0] - image.shape[0]) - pad_top

    pad_left = (pad_size[1] - image.shape[1]) // 2
    pad_right = (pad_size[1] - image.shape[1]) - pad_left

    padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value = (255, 255, 255, 0))

    coord1 = random.choice(range(0, pad_size[0] - target_size[0] + 1))
    coord2 = random.choice(range(0, pad_size[1] - target_size[1] + 1))

    return padded[coord1:coord1+target_size[0], coord2:coord2+target_size[1]]

def diversify_image(image, target_size, max_angle, max_ratio, scale_range, pad_size):
    '''
    Creates a distorted version of the given image by rotating it, modifying its aspect ratio, scaling it, and recropping it.

    Arguments:
        image       -- The image to distort.
        target_size -- The size of the resulting distorted image.
        max_angle   -- The absolute maximum possible rotation angle.
        max_ratio   -- The maximum possible degree of aspect ratio distortion.
        scale_range -- The range of possible image scales.
        pad_size    -- The size the image to padded to before recropping.

    Returns:
        The distorted image.
    '''

    ops = [lambda x : random_rotate(x, max_angle), lambda x : random_ratio(x, max_ratio)]
    random.shuffle(ops)

    for op in ops:
        image = op(image)

    scaled = random_scale(image, target_size, scale_range)
    recropped = random_recrop(scaled, target_size, pad_size)

    return recropped

def diversify_images(in_path, out_path, target_size, max_angle, max_ratio, scale_range, pad_size, diversify_count, extension = ""):
    '''
    Creates a distorted versions of a dataset of images by rotating them, modifying their aspect ratio, scaling them, and recropping them a number of times.

    Arguments:
        in_path         -- The path to the directory from which to load the images.
        out_path        -- The path to the directory to which to save the distorted images.
        target_size     -- The size of the resulting distorted image.
        max_angle       -- The absolute maximum possible rotation angle.
        max_ratio       -- The maximum possible degree of aspect ratio distortion.
        scale_range     -- The range of possible image scales.
        pad_size        -- The size the image to padded to before recropping.
        buffer_size     -- The minimal distance of the crop region from the edge of padded image.
        diversify_count -- The number of distorted images generated for each image of the input set.
        extension       -- The file type extension of images to load.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_UNCHANGED)
        for i in range(diversify_count):
            diversified_image = diversify_image(image, target_size, max_angle, max_ratio, scale_range, pad_size)
            cv2.imwrite(os.path.join(out_path, file_name[0:file_name.rfind(".")] + "_" + str(i) + ".png"), diversified_image)

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
        print("Max angle:\t", end = '')
        argv.append(input())
        print("Max ratio:\t", end = '')
        argv.append(input())
        print("Min scale:\t", end = '')
        argv.append(input())
        print("Max scale:\t", end = '')
        argv.append(input())
        print("Pad height:\t", end = '')
        argv.append(input())
        print("Pad width:\t", end = '')
        argv.append(input())
        print("Buffer height:\t", end = '')
        argv.append(input())
        print("Buffer width:\t", end = '')
        argv.append(input())
        print("Diversify count:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (11, 12):
        target_size = [int(argv[2]), int(argv[3])]
        max_angle = float(argv[4])
        max_ratio = float(argv[5])
        scale_range = [float(argv[6]), float(argv[7])]
        pad_size = [int(argv[8]), int(argv[9])]
        diversify_count = int(argv[10])
        extension = argv[11] if len(argv) == 12 else ""

        diversify_images(argv[0], argv[1], target_size, max_angle, max_ratio, scale_range, pad_size, diversify_count, extension = extension)
    else:
        print("Usage: diversify_images in_path out_path target_height target_width max_angle max_ratio min_scale max_scale pad_height pad_width buffer_height buffer_width diversify_count [extension]")

if __name__ == '__main__':
    main(sys.argv)