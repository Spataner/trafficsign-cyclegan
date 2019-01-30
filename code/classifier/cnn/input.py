'''
CycleGAN input loading operations.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import json
import tensorflow as tf


SHUFFLE_BUFFER_SIZE = 10000
DEG2RAD = 0.017453292519943295


def map_to_image(file_name, channel_count, image_size = None):
    '''
    Mapper function for mapping file names to decoded images.

    Arguments:
        file_name       -- The full path to the image file.
        channel_count   -- The number of image channels.
        image_size      -- A list of two integers specifying height and width of the images. May be None.
                           Used to make sure the image size is known at graph construction time.

    Returns:
        A 3D tensor containing the image.
    '''

    raw_image = tf.read_file(file_name)
    decoded_image = tf.image.decode_png(raw_image, channels = channel_count)
    converted_image = tf.cast(decoded_image, tf.float32)

    if image_size is None:
        target_shape = [tf.shape(converted_image)[0], tf.shape(converted_image)[1], channel_count]
    else:
        target_shape = [image_size[0], image_size[1], channel_count]

    #Reshape operation so at least channel dimensions and optionally even the height and width are known a graph construction time.
    #Also causes early failure if dataset contains different-sized images (which is not supported during training or testing).
    reshaped_image = tf.reshape(converted_image, target_shape)

    return reshaped_image

def augment_image(image, angle_range, scale_range, translation_range, image_size, channel_count):
    '''
    Performs data augmentation on the given image by upscaling and randomly rescropping it as well as through horizontal flipping.

    Arguments:
        image           -- The image to augment with shape [image_height, image_width, channels].
        angle_range         -- The range of angles for the randomized rotation.
        scale_range         -- The range for randomized scaling factors.
        translation_range   -- The range for randomized translation in pixels.
        image_size          -- The target image size.
        channel_count       -- The number of image channels.

    Returns:
        The augmented image.
    '''

    angle = tf.random_uniform([1], minval = angle_range[0], maxval = angle_range[1], dtype = tf.float32) * DEG2RAD
    scale = tf.random_uniform([2], minval = scale_range[0], maxval = scale_range[1], dtype = tf.float32)
    translation = tf.random_uniform([2], minval = translation_range[0], maxval = translation_range[1], dtype = tf.float32)

    transform = [scale[0] * tf.cos(angle[0]), scale[0] * tf.sin(angle[0]), scale[0] * translation[0], -scale[1] * tf.sin(angle[0]), scale[1] * tf.cos(angle[0]), scale[1] * translation[1], 0., 0.]

    return tf.contrib.image.transform(image, transform, interpolation = "BILINEAR")

def create_image_dataset(dataset, batch_size, epoch_count, channel_count, augmenter = None, class_count = -1, image_size = None):
    '''
    Creates a batched and repeated image dataset from a dataset of file names.
    If the input dataset consists of tuples of file names and class labels, the class labels will be converted
    to their one-hot encoding automatically.

    Arguments:
        dataset         -- The unbatched, unrepeated dataset of file names.
        batch_size      -- The number of images per batch.
        epoch_count     -- The number of epochs (repetitions) of the dataset.
        channel_count   -- The number of image channels.
        augmenter       -- An optional dataset augmentation function.
        class_count     -- The number of discriminator classes (for one-hot encoding).
        image_size      -- A list of two integers specifying height and width of the images. May be None.
                           Used to make sure the image size is known at graph construction time.

    Returns:
        The image dataset.
    '''

    if not type(dataset.output_types) == tuple:
        mapper = lambda x : map_to_image(x, channel_count, image_size = image_size)
    else:
        mapper = lambda x, y : (map_to_image(x, channel_count, image_size = image_size), tf.one_hot(y, class_count))

    dataset = dataset.map(mapper)

    if augmenter is not None and image_size is not None:
        dataset = augmenter(dataset)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch_count)

    return dataset

def create_filename_dataset(in_path, shuffle = True, extension = "", labels_path = None):
    '''
    Creates a dataset of all file names (with a certain extension) in the given directory.
    Optionally, integer class labels for each file can be loaded, making the output a dataset of tuples.

    Arguments:
        in_path     -- The path to the directory from which to take the file names.
        shuffle     -- Whether to shuffle the order of the file names.
        extension   -- The file extension the file names must match to be included.
        labels_path -- The path to a JSON file containing a mapping from file name to integer class label.

    Returns:
        The dataset of file names.
    '''

    if not os.path.isdir(in_path):
        raise Exception()

    if labels_path is not None and not os.path.isfile(labels_path):
        raise Exception()
    
    file_names = [
        file_name for file_name in os.listdir(in_path)
        if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)
    ]

    if len(file_names) == 0:
        raise Exception()

    if labels_path is not None:
        with open(labels_path, 'r') as labels_file: labels = json.load(labels_file)
        labels = [labels[file_name] for file_name in file_names]
        label_dataset = tf.data.Dataset.from_tensor_slices(labels)

    file_names = [os.path.join(in_path, file_name) for file_name in file_names]

    dataset = tf.data.Dataset.from_tensor_slices(file_names)

    if labels_path is not None:
        dataset = dataset.zip((dataset, label_dataset))

    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)

    return dataset, len(file_names)

def augment_dataset(dataset, angle_range, scale_range, translation_range, image_size, channel_count):
    '''
    Performs data augmentation on a given dataset of images (through scaling, translation, and rotation).

    Arguments:
        dataset             -- The dataset to augment.
        angle_range         -- The range of angles for the randomized rotation.
        scale_range         -- The range for randomized scaling factors.
        translation_range   -- The range for randomized translation in pixels.
        image_size          -- The target image size.
        channel_count       -- The number of image channels.

    Returns:
        The dataset of augmented images.
    '''

    if not type(dataset.output_types) == tuple:
        mapper = lambda x: augment_image(x, angle_range, scale_range, translation_range, image_size, channel_count)
    else:
        mapper = lambda x, y: (augment_image(x, angle_range, scale_range, translation_range, image_size, channel_count), y)

    return dataset.map(mapper)