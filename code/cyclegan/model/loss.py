'''
CycleGAN training losses.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import tensorflow as tf


DISCRIMINATOR_LOSS_NAME = "discriminator_loss"
MAPPING_LOSS_NAME = "mapping_loss_name"


def discriminator_loss(discriminator, factor = 0.5, input_target = 1.0, mapping_target = 0.0):
    '''
    Creates the loss term for the given discriminator as a sum of square losses on real and generated images.
    Uses a Least-Squares loss.

    Arguments:
        discriminator   -- The object of the discriminator for which to create the loss.
        factor          -- A factor the loss is multiplied with to adjust the learning speed.

    Returns:
        The loss's output tensor.
    '''

    with tf.name_scope(DISCRIMINATOR_LOSS_NAME):
        input_loss = tf.reduce_mean(tf.squared_difference(discriminator.from_input.main, input_target))
        mapping_loss = tf.reduce_mean(tf.squared_difference(discriminator.from_image_buffer.main, mapping_target))

        full_loss = tf.add(input_loss, mapping_loss)

        return tf.multiply(full_loss, factor)

def get_input_target_class(label, class_count):
    '''
    Creates the label-based target vector of a real example for use with a PatchGAN discriminator.

    Argument:
        label       -- The label of the example in one-hot encoding.
        class_count -- The total number of discriminator classes.

    Returns:
        The target vector for the example.
    '''

    return tf.reshape(label, [-1, 1, 1, class_count])

def get_mapping_target_nplusone_class(label, class_count):
    '''
    Creates the label-based target vector for a generated example for use with a PatchGAN discriminator.
    Assumes the discriminator distinguishes N+1 labels (N real data classes and one fake class).

    Arguments:
        label       -- The real label to which the generated example aspires.
        class_count -- The total number of discriminator classes.

    Returns:
        The target vector for the example.
    '''

    return tf.reshape(tf.one_hot(tf.tile([class_count-1], [tf.shape(label)[0]]), class_count), [-1, 1, 1, class_count])

def get_mapping_target_twon_class(label, class_count):
    '''
    Creates the label-based target vector for a generated example for use with a PatchGAN discriminator.
    Assumes the discriminator distinguishes 2N labels (One class each for real and generated variants of each data class).

    Arguments:
        label       -- The real label to which the generated example aspires.
        class_count -- The total number of discriminator classes.

    Returns:
        The target vector for the example.
    '''

    return tf.reshape(tf.concat([label[:, class_count//2:], label[:, :class_count//2]], 1), [-1, 1, 1, class_count])

def discriminator_loss_nplusone_class(discriminator, class_count, factor = 0.5):
    '''
    Creates the loss term for the given discriminator as a sum of square losses on real and generated images.
    Uses a Least-Squares loss.
    Assumes the discriminator distinguishes N+1 labels (N real data classes and one fake class).

    Arguments:
        discriminator   -- The object of the discriminator for which to create the loss.
        class_count     -- The total number of discriminator classes.
        factor          -- A factor the loss is multiplied with to adjust the learning speed.

    Returns:
        The loss's output tensor.
    '''

    input_target = get_input_target_class(discriminator.from_input[1], class_count)
    mapping_target = get_mapping_target_nplusone_class(discriminator.from_image_buffer[1], class_count)

    return discriminator_loss(discriminator, factor = factor, input_target = input_target, mapping_target = mapping_target)

def discriminator_loss_twon_class(discriminator, class_count, factor = 0.5):
    '''
    Creates the loss term for the given discriminator as a sum of square losses on real and generated images.
    Uses a Least-Squares loss.
    Assumes the discriminator distinguishes 2N labels (One class each for real and generated variants of each data class).

    Arguments:
        discriminator   -- The object of the discriminator for which to create the loss.
        class_count     -- The total number of discriminator classes.
        factor          -- A factor the loss is multiplied with to adjust the learning speed.

    Returns:
        The loss's output tensor.
    '''

    input_target = get_input_target_class(discriminator.from_input[1], class_count)
    mapping_target = get_mapping_target_twon_class(discriminator.from_image_buffer[1], class_count)

    return discriminator_loss(discriminator, factor = factor, input_target = input_target, mapping_target = mapping_target)

def info_gan_loss(discriminator):
    '''
    Creates the infoGAN loss term as the mean squared error between the latent variables used in generation and the
    reconstruction thereof by the infoGAN discriminator.

    Arguments:
        discriminator   -- The discriminator for which to calculate the infoGAN loss term.

    Returns:
        The loss's output tensor.
    '''

    return tf.reduce_mean(tf.squared_difference(discriminator.from_mapping[-1], discriminator.from_mapping[-2]))

def mapping_loss(input_x, input_y, cycle_gan, cycle_consistency_factor, adversarial_target_x = 1.0, adversarial_target_y = 1.0):
    '''
    Creates the loss term for the two mappings as a sum of cycle-consistency and adversarial losses.
    Uses L1 norm for the cycle-consistency and a Least-Squares loss for the adversarial loss.

    Arguments:
        input_x                     -- The input of real images for the X domain.
        input_y                     -- The input of real images for the Y domain.
        cycle_gan                   -- The CycleGAN for whose mappings to create the loss.
        cycle_consistency_factor    -- The factor of the cycle-consistency loss in the sum.

    Returns:
        The loss's output tensor
    '''

    with tf.name_scope(MAPPING_LOSS_NAME):
        adversarial_loss_xy = tf.reduce_mean(tf.squared_difference(cycle_gan.discriminator_y.from_mapping.main, adversarial_target_y))
        adversarial_loss_yx = tf.reduce_mean(tf.squared_difference(cycle_gan.discriminator_x.from_mapping.main, adversarial_target_x))

        adversarial_loss = tf.add(adversarial_loss_xy, adversarial_loss_yx)

        cycle_consistency_loss_xy = tf.reduce_mean(tf.abs(tf.subtract(cycle_gan.mapping_xy.cycle.main, input_y)))
        cycle_consistency_loss_yx = tf.reduce_mean(tf.abs(tf.subtract(cycle_gan.mapping_yx.cycle.main, input_x)))

        cycle_consistency_loss = tf.add(cycle_consistency_loss_xy, cycle_consistency_loss_yx)

        return tf.add(adversarial_loss, tf.multiply(cycle_consistency_factor, cycle_consistency_loss))

def mapping_loss_class(input_x, input_y, cycle_gan, cycle_consistency_factor, class_count):
    '''
    Creates the loss term for the two mappings as a sum of cycle-consistency and adversarial losses.
    Assumes the discriminator distinguishes classes.
    Uses L1 norm for the cycle-consistency and a Least-Squares loss for the adversarial loss.

    Arguments:
        input_x                     -- The input of real images for the X domain.
        input_y                     -- The input of real images for the Y domain.
        cycle_gan                   -- The CycleGAN for whose mappings to create the loss.
        cycle_consistency_factor    -- The factor of the cycle-consistency loss in the sum.
        class_count                 -- The total number of discriminator classes.

    Returns:
        The loss's output tensor
    '''

    adversarial_target_x = get_input_target_class(cycle_gan.mapping_yx.from_input[1], class_count)
    adversarial_target_y = get_input_target_class(cycle_gan.mapping_xy.from_input[1], class_count)

    return mapping_loss(
        input_x, input_y, cycle_gan, cycle_consistency_factor, adversarial_target_x = adversarial_target_x,
        adversarial_target_y = adversarial_target_y
    )