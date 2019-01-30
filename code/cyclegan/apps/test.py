'''
Application for running an existing CycleGAN model on a test dataset.
Usage: test in_config_path [name]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import tensorflow                   as tf
import cyclegan.model.architecture  as arch
import cyclegan.util.config         as cfg
import cyclegan.util.exceptions     as exc
import cyclegan.model.output        as output
import cyclegan.model.input         as input


EPOCH_COUNT = 1
BATCH_SIZE = 1


def make_datasets(config):
    '''
    Creates a pair of datasets from a config.

    Arguments:
        config -- The configuration to use for the creation of the datasets.

    Returns:
        One dataset for each domain.
    '''

    if config[cfg.CLASS_MODE] != cfg.CLASS_CONDITIONAL:
        dataset_x, dataset_y, _ = input.create_domain_datasets(
            config[cfg.TEST_CONFIG][cfg.DOMAIN_X_PATH], config[cfg.TEST_CONFIG][cfg.DOMAIN_Y_PATH],
            BATCH_SIZE, EPOCH_COUNT,
            config[cfg.IMAGE_SIZE], config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y],
            extension_x = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_X],
            extension_y = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_Y],
            shuffle = False
        )
    else:
        dataset_x, dataset_y, _ = input.create_domain_datasets(
            config[cfg.TEST_CONFIG][cfg.DOMAIN_X_PATH], config[cfg.TEST_CONFIG][cfg.DOMAIN_Y_PATH],
            BATCH_SIZE, EPOCH_COUNT,
            config[cfg.IMAGE_SIZE], config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y],
            labels_x_path = config[cfg.TEST_CONFIG][cfg.LABELS_X_PATH],
            labels_y_path = config[cfg.TEST_CONFIG][cfg.LABELS_Y_PATH],
            class_count = config[cfg.CLASS_COUNT],
            extension_x = config[cfg.TEST_CONFIG][cfg.EXTENSION_X],
            extension_y = config[cfg.TEST_CONFIG][cfg.EXTENSION_Y],
            shuffle = False
        )

    return dataset_x, dataset_y

def make_inputs(config, dataset):
    '''
    Creates a wrapped input tensor tuple from a configuration and dataset.

    Arguments:
        config      -- The configuration on which to base the tensor tuple.
        dataset     -- The dataset from which to create the tensor tuple.

    Returns:
        The wrapped input tensor tuple.
    '''

    next = dataset.make_one_shot_iterator().get_next()

    if config[cfg.INFO_GAN]:
        info_gan_vector = tf.random_uniform([BATCH_SIZE, config[cfg.INFO_GAN_SIZE]], dtype = tf.float32) * 2 - 1

        if type(next) != tuple:
            next = (next, info_gan_vector)
        else:
            next += (info_gan_vector,)

    return arch.TupleWrapper(next)

def make_classes(config):
    '''
    Chooses appropriate mapping and discriminator classes given a config.

    Arguments:
        config -- The configuration on which to base the decision.

    Returns:
        A mapping and a discriminator class.
    '''

    if config[cfg.CLASS_MODE] == cfg.CLASS_LOSS:
        MappingClass = arch.Mapping
        DiscriminatorClass = arch.Discriminator
    elif config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        MappingClass = arch.get_vector_decorated_mapping([True], [True])
        DiscriminatorClass = arch.get_vector_decorated_discriminator([True], [True])
    elif config[cfg.INFO_GAN]:
        MappingClass = arch.get_vector_decorated_mapping([True], [True])
        DiscriminatorClass = arch.get_info_gan_discriminator(config[cfg.INFO_GAN_SIZE])
    else:
        MappingClass = arch.Mapping
        DiscriminatorClass = arch.Discriminator

    return MappingClass, DiscriminatorClass

def main(argv):
    if len(argv) < 2:
        raise exc.AppException.missing_config()

    print("Started CycleGAN test.")
    print()

    print("Loading configuration...")

    config = cfg.load_config(argv[1], template = cfg.CONFIG_TEMPLATE_TEST)

    cfg.warning(config)

    if len(argv) > 2:
        name = argv[2]
    else:
        name = cfg.retrieve_most_recent_name(config)

    if name is None and cfg.requires_name(config):
        raise exc.AppException.missing_name()

    if name is not None:
        cfg.adjust_paths(config, name, training_paths = False)
        
    cfg.ensure_present(config[cfg.TEST_CONFIG][cfg.OUTPUT_PATH])

    print()
    print("Configuration description:")
    print(config[cfg.DESCRIPTION] if len(config[cfg.DESCRIPTION]) > 0 else "None provided")
    print()

    print("Building graph...")

    tf.reset_default_graph()

    dataset_x, dataset_y, = make_datasets(config)

    input_x = make_inputs(config, dataset_x)
    input_y = make_inputs(config, dataset_y)

    MappingClass, DiscriminatorClass = make_classes(config)

    cycle_gan = arch.CycleGAN(MappingClass, DiscriminatorClass, config, mode = arch.Mode.TEST)
    cycle_gan.create_cycle_gan(input_x.all, input_y.all)

    maximum_channel_count = max([config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y]])

    output_x = output.output_test_image(
        input_x.main, cycle_gan.mapping_xy.from_input.main, cycle_gan.mapping_yx.cycle.main, config[cfg.TEST_CONFIG][cfg.OUTPUT_PATH],
        "test_xyx", target_channel_count = maximum_channel_count
    )
    output_y = output.output_test_image(
        input_y.main, cycle_gan.mapping_yx.from_input.main, cycle_gan.mapping_xy.cycle.main, config[cfg.TEST_CONFIG][cfg.OUTPUT_PATH],
        "test_yxy", target_channel_count = maximum_channel_count
    )

    print("Initializing session...")

    #Modify GPU memory handling to avoid OOM errors.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  #pylint: disable = E1101

    with tf.Session(config = session_config) as session:
        print("Testing...")

        session.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(config[cfg.CHECKPOINT_PATH])

        if checkpoint and checkpoint.model_checkpoint_path:             #pylint: disable = E1101
            saver = tf.train.Saver(cycle_gan.get_mapping_variables())
            saver.restore(session, checkpoint.model_checkpoint_path)    #pylint: disable = E1101
        else:
            print("Unable to restore checkpoint.")
            return

        while True:
            try:
                session.run([output_x, output_y])
            except tf.errors.OutOfRangeError:
                break

    print("All done.")


if __name__ == '__main__':
    tf.app.run()