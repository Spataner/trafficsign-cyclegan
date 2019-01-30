'''
Application for replacing transparent pixels for a collection of images via some randomized scheme.
Usage: add_background in_path out_path generator [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import random
import os
import sys
import cv2
import numpy as np

try:
    import opensimplex
    simplex_available = True
except:
    simplex_available = False


class WhiteNoiseGenerator:
    '''
    Generator class for white RGB noise.
    Determines the color of the RGBA tuple for each postion uniformly at random.
    '''

    def __init__(self, pixel_max):
        self.pixel_max = pixel_max

    def __call__(self, row, column):
        return [random.choice(range(self.pixel_max)), random.choice(range(self.pixel_max)), random.choice(range(self.pixel_max)), self.pixel_max]

class HomogeneousNoiseGenerator:
    '''
    Generator for random but homogeneous background colors.
    Picks a color uniformly at random during construction and returns it at any call.
    '''

    def __init__(self, pixel_max):
        self.color = [random.choice(range(pixel_max)), random.choice(range(pixel_max)), random.choice(range(pixel_max)), pixel_max]

    def __call__(self, row, column):
        return self.color

class SimplexNoiseGenerator:
    '''
    Generator for simplex random noise.
    Picks a color from a 3D simplex noise generator based on the scaled row and column values.
    '''

    def __init__(self, pixel_max, factor):
        self.simplex = opensimplex.OpenSimplex(seed = random.randint(0, 2**32-1))
        self.factor = factor
        self.pixel_max = pixel_max

    def __call__(self, row, column):
        return [
            int((self.simplex.noise3d(self.factor * row, self.factor * column, 0) + 1) * self.pixel_max / 2),
            int((self.simplex.noise3d(self.factor * row, self.factor * column, 1) + 1) * self.pixel_max / 2),
            int((self.simplex.noise3d(self.factor * row, self.factor * column, 2) + 1) * self.pixel_max / 2),
            self.pixel_max
        ]

def string_to_generator(string, pixel_max):
    '''
    Function mapping generator strings to generator callable.

    Arguments:
        string      -- The string specifying the generator to use.
        pixel_max         -- The maximal pixel value.

    Returns:
        A callable that takes two arguments (row and column) and returns an RGBA tuple.
    '''

    if string == "white":
        return WhiteNoiseGenerator(pixel_max)

    if string == "homogeneous":
        return HomogeneousNoiseGenerator(pixel_max)

    if string.startswith("simplex:") and simplex_available:
        try:
            factor = float(string[len("simplex:"):])
            return SimplexNoiseGenerator(pixel_max, factor)
        except:
            pass

    raise Exception("Invalid generator string.")

def add_background(image, generator):
    '''
    Adds a random background to an image in place of any transparent pixels.

    Arguments:
        image           -- The image to which to add noise.
        generator       -- A string specifying the type of generator to use.
    
    Returns:
        The image with added noise and the alpha channel removed.
    '''

    generator = string_to_generator(generator, np.iinfo(image.dtype).max)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j, 3] == 0:
                image[i, j, :] = generator(i, j)

    return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

def add_backgrounds(in_path, out_path, generator, extension = ""):
    '''
    Adds a random background to a collection of images in place of any transparent pixles.

    Arguments:
        in_path     -- The path to the input directory.
        out_path    -- The path to the output directory.
        generator   -- A string specifying the type of generator to use.
        extension   -- The file type extension of the images to load.
    '''

    file_names = [file_name for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]

    for file_name in file_names:
        image = cv2.imread(os.path.join(in_path, file_name), cv2.IMREAD_UNCHANGED)
        noisy_image = add_background(image, generator)
        cv2.imwrite(os.path.join(out_path, file_name), noisy_image)

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())
        print("Output path:\t", end = '')
        argv.append(input())
        print("Generator:\t", end = '')
        argv.append(input())
        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (3, 4):
        add_backgrounds(argv[0], argv[1], argv[2], extension = argv[3] if len(argv) > 3 else "")
    else:
        print("Usage: add_background in_path out_path generator [extension]")


if __name__ == "__main__":
    main(sys.argv)