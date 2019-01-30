'''
CycleGAN training operations.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import tensorflow as tf


LEARNING_RATE_NAME = "learning_rate"
DISCRIMINATOR_TRAINING_OP_NAME = "discriminator_training_op"
MAPPING_TRAINING_OP_NAME = "mapping_training_op"


def create_delayed_linear_decay(learning_rate, constant_epoch_count, linear_epoch_count, epoch_batch_count):
    '''
    Creates a delayed linear learning rate decay.

    Arguments:
        learning_rate           -- The initial (undecayed) learning rate.
        constant_epoch_count    -- The number of epochs for which the learning rate remains constant.
        linear_epoch_count      -- The number of epochs over which the learning rate linearly decays to zero.
        epoch_batch_count       -- The number of batches per epoch.

    Returns:
        The learning rate's tensor.
    '''

    with tf.name_scope(LEARNING_RATE_NAME):
        step = tf.Variable(initial_value = -1, trainable = False)
        incremented_step = tf.assign_add(step, 1)

        epoch = tf.floordiv(incremented_step, epoch_batch_count)
        decay_epoch = linear_epoch_count - epoch + constant_epoch_count
        decay_delta = decay_epoch / linear_epoch_count

        decayed_learning_rate = tf.minimum(learning_rate, tf.cast(decay_delta * learning_rate, dtype = tf.float32))

        return decayed_learning_rate

def create_discriminator_training_op(discriminator, loss, learning_rate, beta1, beta2):
    '''
    Creates the Adam training operation for a discriminator.

    Arguments:
        discriminator   -- The discriminator for which to create the training operation.
        loss            -- The discriminator's loss.
        learning_rate   -- The learning rate for the optimizer.
        beta1           -- beta1 parameter for the Adam optimizer.
        beta2           -- beta2 parameter for the Adam optimizer.

    Returns:
        The training operation.
    '''

    with tf.name_scope(DISCRIMINATOR_TRAINING_OP_NAME):
        adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
        gradients = adam.compute_gradients(loss, var_list = discriminator.get_variables())

        return adam.apply_gradients(gradients, global_step = tf.train.get_global_step())

def create_mapping_training_op(cycle_gan, loss, learning_rate, beta1, beta2):
    '''
    Creates the Adam training operation for both mappings of a CycleGAN.

    Arguments:
        cycle_gan       -- The CycleGAN for whose mappings to create the training operation.
        loss            -- The mappings' loss.
        learning_rate   -- The learning rate for the optimizer.
        beta1           -- beta1 parameter for the Adam optimizer.
        beta2           -- beta2 parameter for the Adam optimizer.

    Returns:
        The training operations.
    '''

    with tf.name_scope(MAPPING_TRAINING_OP_NAME):
        adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2)

        gradients = adam.compute_gradients(loss, var_list = cycle_gan.get_mapping_variables())

        return adam.apply_gradients(gradients, global_step = tf.train.get_global_step())


def create_training_ops(cycle_gan, mapping_loss, discriminator_loss_x, discriminator_loss_y, learning_rate, beta1, beta2):
    '''
    Creates the three training operations for a CycleGAN.

    Arguments:
        cycle_gan               -- The CycleGAN for which to create the training operations.
        mapping_loss            -- The loss of the CycleGAN's mappings.
        discriminator_loss_x    -- The loss of the CycleGAN's domain X discriminator.
        discriminator_loss_y    -- The loss of the CycleGAN's domain Y discriminator.
        learning_rate           -- The learning rate for the optimizer.
        beta1                   -- The beta1 parameter of the Adam optimizer.
        beta2                   -- The beta2 parameter of the Adam optimizer.

    Returns:
        The training operations for mappings, X domain discriminator, and Y domain discriminator.
    '''

    mapping_training_op = create_mapping_training_op(cycle_gan, mapping_loss, learning_rate, beta1, beta2)

    discriminator_x_training_op = create_discriminator_training_op(cycle_gan.discriminator_x, discriminator_loss_x, learning_rate, beta1, beta2)
    discriminator_y_training_op = create_discriminator_training_op(cycle_gan.discriminator_y, discriminator_loss_y, learning_rate, beta1, beta2)

    return mapping_training_op, discriminator_x_training_op, discriminator_y_training_op