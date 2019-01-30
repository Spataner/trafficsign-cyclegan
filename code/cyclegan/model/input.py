'''
CycleGAN input loading operations.

Notice: Class label annotations are expected to be provided in the form of JSON files containing a single dictionary
which maps an image file name's tail to an integer, like so
{
    "someimage.png": 3,
    "anotherimage.png": 6,
    "yetanotherimage.png": 2,
    [...]
}

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import json
import tensorflow               as tf
import cyclegan.util.exceptions as exc


SHUFFLE_BUFFER_SIZE = 10000


def convert_image(image):
    '''
    Rescales an image to [-1, 1] (to match the generator's output range).

    Arguments:
        image   -- The image tensor to convert.

    Returns:
        The converted image.
    '''

    return tf.subtract(tf.divide(tf.cast(image, tf.float32), 127.5), 1.0)

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

    #We use decode_png rather than decode_image, since they are actually both wrappers for the same underlying function.
    #The only difference is that with decode_png the channels keyword actually works as expected.
    decoded_image = tf.image.decode_png(raw_image, channels = channel_count)
    converted_image = convert_image(decoded_image)

    if image_size is None:
        target_shape = [tf.shape(converted_image)[0], tf.shape(converted_image)[1], channel_count]
    else:
        target_shape = [image_size[0], image_size[1], channel_count]

    #Reshape operation so at least channel dimensions and optionally even the height and width are known a graph construction time.
    #Also causes early failure if dataset contains different-sized images (which is not supported during training or testing).
    reshaped_image = tf.reshape(converted_image, target_shape)

    return reshaped_image

def augment_image(image, resize_shape, image_size, channel_count, flip = True, random_resize = False):
    '''
    Performs data augmentation on the given image by upscaling and randomly rescropping it as well as through horizontal flipping.

    Arguments:
        image           -- The image to augment with shape [image_height, image_width, channels].
        resize_shape    -- The image size for upscaling the image.
        image_size      -- The desired image output size.
        channel_count   -- The number of image channels.
        flip            -- Whether to randomly apply horizontal flipping.
        random_resize   -- Whether the true resize shape should be uniformly drawn from in between the given resize
                           shape and the image size.

    Returns:
        The augmented image.
    '''

    if random_resize:
        resize_shape = [
            tf.random_uniform([1], minval = image_size[0], maxval = resize_shape[0]+1, dtype = tf.int32)[0],
            tf.random_uniform([1], minval = image_size[1], maxval = resize_shape[1]+1, dtype = tf.int32)[0]
        ]

    resized = tf.image.resize_images(image, resize_shape)
    cropped = tf.random_crop(resized, [image_size[0], image_size[1], channel_count])

    if flip:
        output = tf.image.random_flip_left_right(cropped)
    else:
        output = cropped

    return output

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
        mapper = lambda image : map_to_image(image, channel_count, image_size = image_size)
    else:
        mapper = lambda image, label : (map_to_image(image, channel_count, image_size = image_size), tf.one_hot(label, class_count))

    dataset = dataset.map(mapper)

    if augmenter is not None and image_size is not None:
        dataset = augmenter(dataset)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch_count)

    return dataset

def align_datasets(datasets, lengths):
    '''
    Aligns a number of datasets to have the same length by padding shorter datasets to the length of the longest one.

    Arguments:
        datasets    -- A list/tuple of datasets.
        lengths     -- The list/tuple of dataset lengths (with index correspondence).

    Returns:
        A list of datasets of same length as well as an integer specifiying that length.
    '''

    datasets = list(datasets)
    max_length = max(lengths)

    for i in range(len(datasets)):
        datasets[i] = datasets[i].concatenate(datasets[i].take(max_length - lengths[i]))

    return datasets, max_length

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
        raise exc.InputException.invalid_input_path()

    if labels_path is not None and not os.path.isfile(labels_path):
        raise exc.InputException.invalid_labels_path()
    
    file_names = [
        file_name for file_name in os.listdir(in_path)
        if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)
    ]

    if len(file_names) == 0:
        raise exc.InputException.empty_directory()

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

def augment_dataset(dataset, resize_shape, image_size, channel_count, flip = True, random_resize = False):
    '''
    Performs data augmentation on a given dataset of images (through upscaling and random rescropping as well as flipping).

    Arguments:
        dataset         -- The dataset to augment.
        resize_shape    -- The shape the images are resized to before recropping.
        image_size      -- The desired image output size.
        channel_count   -- The number of channels of the images.
        flip            -- Whether to perform random horizontal flipping.
        random_resize   -- Whether the true resize shape should be uniformly drawn from in between the given resize
                           shape and the image size.

    Returns:
        The dataset of augmented images.
    '''

    if not type(dataset.output_types) == tuple:
        mapper = lambda x: augment_image(x, resize_shape, image_size, channel_count, flip = flip, random_resize = random_resize)
    else:
        mapper = lambda x, y: (augment_image(x, resize_shape, image_size, channel_count, flip = flip, random_resize = random_resize), y)

    return dataset.map(mapper)

def create_domain_datasets(
        domain_x_path, domain_y_path, batch_size, epoch_count, image_size, channel_count_x, channel_count_y,
        shuffle = True, extension_x = "", extension_y = "", augmenter = None, labels_x_path = None, labels_y_path = None,
        class_count = -1
    ):
    '''
    Creates a dataset of images for each domain (X and Y). The datasets are aligned, batched, and repeated.

    Arguments:
        domain_x_path   -- The path to the directory holding the images for domain X.
        domain_y_path   -- The path to the directory holding the images for domain Y.
        batch_size      -- The number of images per batch.
        epoch_count     -- The number of epochs (repetitions) for the datasets.
        image_size      -- The size of the images in the domains.
        channel_count_x -- The number of channels for images in domain X.
        channel_count_y -- The number of channels for images in domain Y.
        shuffle         -- Whether to shuffle the datasets.
        extension_x     -- The file type extension for images of domain X.
        extension_y     -- The file type extension for images of domain Y.
        augmenter       -- An optional dataset augmentation function (with two arguments: input, channel_count).
        labels_x_path   -- The path to the label JSON file for images from domain X.
        labels_y_path   -- The path to the label JSON file for images from domain Y.
        class_count     -- The number of classes in the data.

    Returns:
        The two datasets as well as the number of images (not batches!) per epoch.
    '''

    dataset_x, length_x = create_filename_dataset(
        domain_x_path, shuffle = shuffle, extension = extension_x,
        labels_path = labels_x_path
    )

    dataset_y, length_y = create_filename_dataset(
        domain_y_path, shuffle = shuffle, extension = extension_y,
        labels_path = labels_y_path
    )

    (dataset_x, dataset_y), epoch_image_count = align_datasets((dataset_x, dataset_y), (length_x, length_y))

    dataset_x = create_image_dataset(
        dataset_x, batch_size, epoch_count, channel_count_x, image_size = image_size,
        augmenter = (lambda x: augmenter(x, channel_count_x)) if augmenter != None else None,
        class_count = class_count
    )
    
    dataset_y = create_image_dataset(
        dataset_y, batch_size, epoch_count, channel_count_y, image_size = image_size,
        augmenter = (lambda x: augmenter(x, channel_count_y)) if augmenter != None else None,
        class_count = class_count
    )

    return dataset_x, dataset_y, epoch_image_count