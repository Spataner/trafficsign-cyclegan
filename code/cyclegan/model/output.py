'''
CycleGAN output operations.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import math
import tensorflow               as tf
import cyclegan.util.exceptions as exc


PATH_SEPARATOR = "/"
FILE_EXTENSION = ".png"
ENCODER_FUNCTION = tf.image.encode_png
COLLAGE_AXIS = 2
BATCH_STACK_AXIS = 0

TEST_OUTPUT_NAME = "test_output"
TRANSLATED_OUTPUT_NAME = "translated_output"


def convert_image(image):
    '''
    Rescales and converts an image from pixel range [-1, 1] to standard 8-bit.

    Arguments:
        image   -- The image tensor to convert.

    Returns:
        The converted image.
    '''

    return tf.cast(tf.multiply(tf.add(image, 1.0), 127.5), tf.uint8)

def write_image(image, file_name):
    '''
    Writes the given image to file.

    Arguments:
        image       -- The image to write to file.
        file_name   -- The full file name for the file to write.

    Returns:
        The write file output node.
    '''

    target_shape = [tf.shape(image)[1], tf.shape(image)[2], int(image.shape[3])]
    target_shape[BATCH_STACK_AXIS] = -1

    converted_image = tf.reshape(convert_image(image), target_shape)
    raw_image = ENCODER_FUNCTION(converted_image)

    return tf.write_file(file_name, raw_image)

def concatenate_images(input, translated_input, reconstructed_input, target_channel_count = 3):
    '''
    Creates a collage of input, its translation, and its reconstruction.
    Ensures that the images have the same number of channels via tiling and slicing.

    Arguments:
        input                   -- The original input image.
        translated_input        -- The domain-translated image.
        reconstructed_input     -- The reconstructed (cyclicly translated) image.
        target_channel_count    -- The desired number of channels for the collage image.

    Returns:
        The concatenated collage image.
    '''

    
    duplicates_input = math.ceil(target_channel_count / int(input.shape[3]))
    duplicates_translated_input = math.ceil(target_channel_count / int(translated_input.shape[3]))
    duplicates_reconstructed_input = math.ceil(target_channel_count / int(reconstructed_input.shape[3]))

    #Via tiling, ensure that each image has at least target_channel_count channels.
    duplicated_input = tf.tile(input, [1, 1, 1, duplicates_input])
    duplicated_translated_input = tf.tile(translated_input, [1, 1, 1, duplicates_translated_input])
    duplicated_reconstructed_input = tf.tile(reconstructed_input, [1, 1, 1, duplicates_reconstructed_input])

    #Via slicing, ensure that each image has at most target_channel_count channels.
    concatenated_images = tf.concat(
        [duplicated_input[:, :, :, 0:target_channel_count], duplicated_translated_input[:, :, :, 0:target_channel_count],
        duplicated_reconstructed_input[:, :, :, 0:target_channel_count]], COLLAGE_AXIS
    )
    
    return concatenated_images

def output_test_image(input, translated_input, reconstructed_input, output_path, prefix, target_channel_count = 3):
    '''
    Output operation for writing a collage of input image, its translation, and its reconstruction to file.

    Arguments:
        input                   -- The original input image.
        translated_input        -- The domain-translated image.
        reconstructed_input     -- The reconstructed (cyclicly translated) image.
        output_path             -- The path of the output directory.
        prefix                  -- The prefix to which a number is concatenated in order to create unique file names.
        target_channel_count    -- The desired number of channels for the collage image.

    Returns:
        The write file output node.
    '''

    if not os.path.isdir(output_path):
        raise exc.OutputException.invalid_path()

    with tf.name_scope(TEST_OUTPUT_NAME):
        count = tf.Variable(initial_value = 0, dtype = tf.int32, trainable = False)
        incremented_count = tf.assign_add(count, 1)

        concatenated_images = concatenate_images(input, translated_input, reconstructed_input, target_channel_count = target_channel_count)
        
        return write_image(concatenated_images, tf.string_join([output_path, prefix + tf.as_string(incremented_count) + FILE_EXTENSION], separator = PATH_SEPARATOR))

def output_translated_image(input, output_path, prefix):
    '''
    Output operation for writing a translated image to file.

    Arguments:
        input       -- The image to write to file.
        output_path -- The path to the output directory.
        prefix      -- The prefix to which a number is concatenated to create unique file names.

    Returns:
        The write file output node as well as the head of the used file name.
    '''

    if not os.path.isdir(output_path):
        raise exc.OutputException.invalid_path()

    with tf.name_scope(TRANSLATED_OUTPUT_NAME):
        count = tf.Variable(initial_value = 0, dtype = tf.int32, trainable = False)
        incremented_count = tf.assign_add(count, 1)

        file_name_head = prefix + tf.as_string(incremented_count) + FILE_EXTENSION
        file_name = tf.string_join([output_path, file_name_head], separator = PATH_SEPARATOR)
        written_image = write_image(input, file_name)

        return written_image, file_name_head