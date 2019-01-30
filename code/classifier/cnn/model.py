'''
Model specification of a multi-class CNN classifier.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import tensorflow as tf


def conv_layer(input, filter_count, kernel_size):
    '''
    Standard 2D convolution, relu, max-pooling layer.

    Arguments:
        input           -- The input volume.
        filter_count    -- The number of filters in the convolutional layer.
        kernel_size     -- The size of the convolution filters.

    Returns:
        The layer's output node.
    '''

    conv = tf.layers.conv2d(input, filter_count, kernel_size, padding = "same", kernel_initializer = tf.random_uniform_initializer(minval = -0.05, maxval = 0.05), activation = tf.nn.relu)
    max_pool = tf.layers.max_pooling2d(conv, [2, 2], [2, 2])

    return max_pool

def make_classifier(input):
    '''
    Builds the multi-class classifier's graph.

    Arguments:
        input   -- The input images.
    
    Returns:
        The predicted labels for the input.
    '''

    conv1 = conv_layer(input, 100, [7, 7])
    conv2 = conv_layer(conv1, 150, [4, 4])
    conv3 = conv_layer(conv2, 250, [4, 4])

    flattened = tf.layers.flatten(conv3)

    dense1 = tf.layers.dense(flattened, 300, activation = tf.nn.relu, kernel_initializer = tf.random_uniform_initializer(minval = -0.05, maxval = 0.05))
    dense2 = tf.layers.dense(dense1, 43, activation = None, kernel_initializer = tf.random_uniform_initializer(minval = -0.05, maxval = 0.05))

    return dense2

def loss(labels, predicted_labels):
    '''
    Creates the classifier's loss.

    Arguments:
        labels              -- The true labels of the images.
        predicted_labels    -- The labels predicted by the classifier.

    Returns:
        The loss value.
    '''

    return tf.losses.softmax_cross_entropy(labels, predicted_labels)

def optimizer(loss):
    '''
    Creates the optimizer for the classifier.

    Arguments:
        loss    -- The classifier's loss.

    Returns:
        The optimizer's minimization operation.
    '''

    opt = tf.train.AdamOptimizer()
    return opt.minimize(loss, global_step = tf.train.get_or_create_global_step())

def inverse_one_hot(label):
    '''
    Converts a batch of one-hot encoded label vectors to a batch of integer class-labels.

    Arguments:
        label -- A tensor of shape [batch_size, class_count].
    
    Returns:
        A tensor of shape [batch_size].
    '''

    return tf.where(tf.not_equal(label, 0))[:, 1]