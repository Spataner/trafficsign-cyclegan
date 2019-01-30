'''
CycleGAN operations and network layers.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import tensorflow as tf


CONV2D_NAME = "conv2d"
CONV2D_DOWNSAMPLE_NAME = "conv2d_downsample"
CONV2D_UPSAMPLE_NAME = "conv2d_upsample"
CONV2d_DOWNSAMPLE_LEAKY_NAME = "conv2d_downsample_leaky"
RESIDUAL_BLOCK_NAME = "residual_block"
IMAGE_BUFFER_NAME = "image_buffer"
EXTRACT_PATCHES_NAME = "extract_patches"


def instance_normalization(input, epsilon = 1e-6, affine = True):
    '''
    Instance normalization for tensors of shape [batch_size, image_height, image_width, channels].
    As according to Ulyanov et al. (https://arxiv.org/pdf/1607.08022.pdf).

    Arguments:
        input   -- The input tensor of shape [batch_size, image_height, image_width, channels].
        epsilon -- The constant added onto the variance during normalization to avoid divisions by zero.
        affine  -- Whether to add a trainable affine transformation during normalization.

    Returns:
        The image-channel-wise contrast normalized tensor.
    '''

    return tf.contrib.layers.instance_norm(input, epsilon = epsilon, scale = affine, center = affine)   #pylint: disable=E1101

def get_pad_amounts_construction(input, kernel_size, strides):
    '''
    Computes the padding amounts (height and width) for the same padding algorithm at graph construction time.

    Arguments:
        input       -- The input for which to compute the paddings.
        kernel_size -- The size of the convolution kernel.
        strides     -- The convolution stride.

    Returns:
        A tuple of integers the total amount of padding for height and width.
    '''

    if (int(input.shape[1]) % strides[0] == 0):
        pad_along_height = max(kernel_size[0] - strides[0], 0)
    else:
        pad_along_height = max(kernel_size[0] - (int(input.shape[1]) % strides[0]), 0)

    if (int(input.shape[2]) % strides[1] == 0):
        pad_along_width = max(kernel_size[1] - strides[1], 0)
    else:
        pad_along_width = max(kernel_size[1] - (int(input.shape[2]) % strides[1]), 0)

    return pad_along_height, pad_along_width

def get_pad_amounts_execution(input, kernel_size, strides):
    '''
    Computes the padding amounts (height and width) for the same padding algorithm at graph execution time.

    Arguments:
        input       -- The input for which to compute the paddings.
        kernel_size -- The size of the convolution kernel.
        strides     -- The convolution stride.

    Returns:
        A tuple of integer tensors the total amount of padding for height and width.
    '''

    pad_along_height_exact = max(kernel_size[0] - strides[0], 0)
    pad_along_height_remainder = tf.maximum(0, kernel_size[0] - tf.shape(input)[1] % strides[0])

    is_height_exact = tf.equal(tf.shape(input)[1] % strides[0], 0)

    pad_along_height = tf.where(is_height_exact, x = pad_along_height_exact, y = pad_along_height_remainder)

    pad_along_width_exact = max(kernel_size[1] - strides[1], 0)
    pad_along_width_remainder = tf.maximum(0, kernel_size[1] - tf.shape(input)[2] % strides[1])

    is_width_exact = tf.equal(tf.shape(input)[1] % strides[1], 0)

    pad_along_width = tf.where(is_width_exact, x = pad_along_width_exact, y = pad_along_width_remainder)

    return pad_along_height, pad_along_width

def pad_same(input, kernel_size, strides = [1, 1], padding_mode = 'REFLECT'):
    '''
    Reimplementation of Tensorflow's 'SAME' padding algorithm for arbitrary padding schemes.
    As according to https://www.tensorflow.org/api_guides/python/nn#Convolution.

    Arguments:
        input           -- The input image tensor of shape [batch_size, image_height, image_width, channels].
        kernel_size     -- A list of two integers specifying convolution filter size.
        strides         -- A list of two integers specifying convolution stride.
        padding_mode    -- The padding scheme. May be any padding scheme supported by tf.pad.

    Returns:
        The tensor of padded images.
    '''

    #Depending on whether image size is known at construction time, calculate the padding amounts statically or dynamically.

    if input.shape[1].value is None or input.shape[2].value is None:
        pad_along_height, pad_along_width = get_pad_amounts_execution(input, kernel_size, strides)
    else:
        pad_along_height, pad_along_width = get_pad_amounts_construction(input, kernel_size, strides)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    padded = tf.pad(input, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode = padding_mode)

    #If we calculated the padding amounts dynamically, perform a reshape so the channel dimension is again known statically.
    if input.shape[1].value is None or input.shape[2].value is None:
        padded = tf.reshape(padded, [tf.shape(input)[0], tf.shape(input)[1] + pad_along_height, tf.shape(input)[2] + pad_along_width, input.shape[3]])

    return padded

def conv2d_downsample(input, kernel_size, filter_count, initializer_std = 0.02, padding_mode = 'REFLECT', instance_norm_affine = True):
    '''
    Strided Convolution-InstanceNorm-ReLU layer for downsampling.

    Arguments:
        input                   -- The input tensor of shape [batch_size, image_height, image_width, channels].
        kernel_size             -- An integer list of length two specifying convolution filter size.
        filter_count            -- The number of convolution filters for this layer.
        initializer_std         -- The standard deviation of the Gaussian distribution used for weight initialization.
        instance_norm_affine    -- Whether the instance norm shall include a trainable affine transform.

    Returns:
        The output tensor of the layer.
    '''

    with tf.name_scope(CONV2D_DOWNSAMPLE_NAME):
        padded = pad_same(input, kernel_size, strides = [2, 2], padding_mode = padding_mode)
        conv = tf.layers.conv2d(padded, filter_count, kernel_size, strides = 2, padding = 'VALID', kernel_initializer = tf.truncated_normal_initializer(stddev = initializer_std))
        norm = instance_normalization(conv, affine = instance_norm_affine)
        relu = tf.nn.relu(norm)

        return relu

def conv2d_downsample_leaky(input, kernel_size, filter_count, slope, instance_norm = True, initializer_std = 0.02, padding_mode = 'REFLECT', instance_norm_affine = True):
    '''
    Strided Convolution-(InstanceNorm)-LeakyReLU layer for downsampling.

    Arguments:
        input                   -- The input tensor of shape [batch_size, image_height, image_width, channels].
        kernel_size             -- An integer list of length two specifying convolution filter size.
        filter_count            -- The number of convolution filters for this layer.
        slope                   -- The slope (or alpha value) of the leaky ReLU.
        instance_norm           -- Whether to perform instance normalization.
        initializer_std         -- The standard deviation of the Gaussian distribution used for weight initialization.
        instance_norm_affine    -- Whether the instance norm shall include a trainable affine transform.

    Returns:
        The output tensor of the layer.
    '''

    with tf.name_scope(CONV2d_DOWNSAMPLE_LEAKY_NAME):
        padded = pad_same(input, kernel_size, strides = [2, 2], padding_mode = padding_mode)
        conv = tf.layers.conv2d(padded, filter_count, kernel_size, strides = 2, padding = 'VALID', kernel_initializer = tf.truncated_normal_initializer(stddev = initializer_std))

        if instance_norm:
            norm = instance_normalization(conv, affine = instance_norm_affine)
        else:
            norm = conv
            
        relu = tf.nn.leaky_relu(norm, alpha = slope)

        return relu


def conv2d(input, kernel_size, filter_count, initializer_std = 0.02, padding_mode = 'REFLECT', instance_norm = True, activation = tf.nn.relu, instance_norm_affine = True):
    '''
    Convolution-(InstanceNorm) layer.

    Arguments:
        input                   -- The input tensor of shape [batch_size, image_height, image_width, channels].
        kernel_size             -- An integer list of length two specifying convolution filter size.
        filter_count            -- The number of convolution filters for this layer.
        initializer_std         -- The standard deviation of the Gaussian distribution used for weight initialization.
        activation              -- The activation function to use (may be None).
        instance_norm_affine    -- Whether the instance norm shall include a trainable affine transform.

    Returns:
        The output tensor of the layer.
    '''

    with tf.name_scope(CONV2D_NAME):
        padded = pad_same(input, kernel_size, padding_mode = padding_mode)
        conv = tf.layers.conv2d(padded, filter_count, kernel_size, padding = 'VALID', kernel_initializer = tf.truncated_normal_initializer(stddev = initializer_std))

        if instance_norm:
            norm = instance_normalization(conv, affine = instance_norm_affine)
        else:
            norm = conv

        if activation is not None:
            output = activation(norm)
        else:
            output = norm

        return output

def conv2d_upsample(input, kernel_size, filter_count, initializer_std = 0.02, instance_norm_affine = True):
    '''
    Fractionally strided Convolution-InstanceNorm-ReLU layer for upsampling.

    Arguments:
        input                   -- The input tensor of shape [batch_size, image_height, image_width, channels].
        kernel_size             -- An integer list of length two specifying convolution filter size.
        filter_count            -- The number of convolution filters for this layer.
        initializer_std         -- The standard deviation of the Gaussian distribution used for weight initialization.
        instance_norm_affine    -- Whether the instance norm shall include a trainable affine transform.

    Returns:
        The output tensor of the layer.
    '''
    with tf.name_scope(CONV2D_UPSAMPLE_NAME):
        conv = tf.layers.conv2d_transpose(input, filter_count, kernel_size, strides = 2, padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = initializer_std))
        norm = instance_normalization(conv, affine = instance_norm_affine)
        relu = tf.nn.relu(norm)

        return relu


def residual_block(input, layers, activation = tf.nn.relu):
    '''
    Residual layer block for arbitrary layers.
    Note that the last layer's output must have the same shape as the input.
    As according to He et al. (http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).

    Arguments:
        input       -- The input to the block.
        layers      -- A list of single argument functions for constructing the block's layers.
        activation  -- The activation function applied after adding the last layer's output to the block's input (may be None).

    Returns:
        The residual block's output tensor with the same shape as the input.
    '''

    with tf.name_scope(RESIDUAL_BLOCK_NAME):
        output = input

        for layer in layers:
            output = layer(output)

        sum = tf.add(input, output)

        if activation is not None:
            output = activation(sum)
        else:
            output = sum

        return output

def tensor_buffer(input, buffer_size):
    '''
    Tensor buffer layer.
    This layer provides, as output, an image batch with the same size as the input image batch.
    Each time output is requested, the layer itself requests a new image batch from its input. It then randomly decides for every image in the batch
    whether to output that image directly or to store the image in the buffer. In the latter case, the buffered image that is replaced is
    output instead.

    Arguments:
        input       -- The image buffer's image source.
        buffer_size -- The maximal capacity of the image buffer.

    Returns:
        The buffer's output node.
    '''

    if not type(input) == tuple:
        inputs = tuple([input])
    else:
        inputs = input

    with tf.name_scope(IMAGE_BUFFER_NAME):
        count = tf.Variable(initial_value = 0, dtype = tf.int32, trainable = False)

        batch_size = tf.shape(inputs[0])[0]
        incremented_count = tf.assign(count, tf.add(count, batch_size))
        old_count = incremented_count - batch_size

        #The position that we replace is either random (if the buffer is at maximal capacity) or the smallest empty position.

        sequential_indices = tf.range(old_count, incremented_count, dtype = tf.int32)

        random_indices = tf.random_shuffle(tf.range(0, limit = tf.minimum(old_count, buffer_size), dtype = tf.int32))
        random_indices = tf.concat([random_indices, tf.zeros([buffer_size], dtype = tf.int32)], 0)
        random_indices = random_indices[:batch_size]

        unfilled_count = tf.minimum(tf.maximum(0, buffer_size - old_count), batch_size)

        index_mask = tf.concat([tf.ones(unfilled_count, dtype = tf.int32), tf.zeros(batch_size - unfilled_count, dtype = tf.int32)], 0)

        replaced_indices = tf.where(tf.equal(index_mask, 1), x = sequential_indices, y = random_indices)

        #If the buffer is not at maximal capacity, store the input images in the buffer and output them.
        #Otherwise, randomly decide whether to output the input directly or whether to store the input and output the image it replaces.

        use_buffer = tf.random_uniform([batch_size], minval = 0, maxval = 2, dtype = tf.int32)
        store_input = tf.logical_or(tf.equal(use_buffer, 1), tf.equal(index_mask, 1))
        output_input = tf.logical_or(tf.equal(index_mask, 1), tf.equal(use_buffer, 0))

        outputs = list()

        for input_element in inputs:
            buffer = tf.Variable(
                initial_value = tf.zeros([buffer_size] + [int(input_element.shape[i]) for i in range(1, len(input_element.shape))]),
                dtype = tf.float32, trainable = False
            )

            buffer_section = tf.gather(buffer, replaced_indices)

            output = tf.where(output_input, x = input_element, y = buffer_section)

            with tf.control_dependencies([output]):
                assigned_buffer = tf.scatter_update(buffer, replaced_indices, tf.where(store_input, input_element, buffer_section))

            with tf.control_dependencies([assigned_buffer]):
                outputs.append(tf.identity(output))

        if type(input) == tuple:
            return outputs
        
        return outputs[0]


def extract_patches_to_batch(input, patch_size, strides = [1, 1]):
    '''
    Extracts patches from an input tensor of shape [batch_size, image_height, image_width, channels] and stacks them in the batch dimension.

    Arguments:
        input       -- The input image batch tensor.
        patch_size  -- A list of two integers specifying height and width of the extracted patches.
        strides     -- A list of two integers specifying vertical offset between centers of consecutive patches of the same column and
                       horizontal offset between centers of consecutive patches of the same row.
    
    Returns:
        A tensor of shape [batch_size * patch_count, patch_height, patch_width, channels], where patches from the same input image are consecutive in the first dimension.
    '''

    with tf.name_scope(EXTRACT_PATCHES_NAME):
        rates = [1, 1, 1, 1]
        ksizes = [1, patch_size[0], patch_size[1], 1]
        strides = [1, strides[0], strides[1], 1]

        patches = tf.extract_image_patches(input, ksizes = ksizes, strides = strides, rates = rates, padding = 'VALID')
        patches_reshaped = tf.reshape(patches, [tf.shape(patches)[0] * (int(patches.shape[1]) * int(patches.shape[2])), patch_size[0], patch_size[1], int(input.shape[3])])

        return patches_reshaped

def inverse_one_hot(label):
    '''
    Converts a batch of one-hot encoded label vectors to a batch of integer class-labels.

    Arguments:
        label -- A tensor of shape [batch_size, class_count].
    
    Returns:
        A tensor of shape [batch_size].
    '''

    return tf.where(tf.not_equal(label, 0))[:, 1]