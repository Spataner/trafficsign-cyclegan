'''
Application for running an existing CycleGAN model on a dataset (unidirectionally).
Usage: run in_config_path direction in_path out_path [-e extension] [-i in_labels_path] [-o out_labels_path] [-n model_name]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import json
import tensorflow                   as tf
import cyclegan.model.architecture  as arch
import cyclegan.util.config         as cfg
import cyclegan.util.exceptions     as exc
import cyclegan.model.output        as output
import cyclegan.model.input         as input
import cyclegan.model.operations    as ops
import os


EPOCH_COUNT = 1
BATCH_SIZE = 1
XY_DIRECTION = "xy"
YX_DIRECTION = "yx"


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
        info_gan_vector = tf.random_uniform([config[cfg.INFO_GAN_SIZE]], dtype = tf.float32) * 2 - 1

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

def parse_args(argv):
    '''
    Parses an argument list, filtering flag-value pairs out into a separate dictionary.

    Arguments:
        argv -- The raw argument list.

    Returns:
        A list of actual arguments and a dictionary of flags and their values.
    '''

    arguments = list()
    key_words = dict()

    i = 0
    while i < len(argv):
        if argv[i][0] == "-" and i < len(argv)-1:
            key_words[argv[i]] = argv[i+1]
            i += 1
        else:
            arguments.append(argv[i])
        i += 1

    return arguments, key_words

def main(argv):
    arguments, key_words = parse_args(argv)

    if len(arguments) < 5:
        raise exc.AppException.missing_arguments()

    print("Started CycleGAN run.")
    print()

    print("Loading configuration...")

    config = cfg.load_config(arguments[1], template = cfg.CONFIG_TEMPLATE_BASE)
    direction = arguments[2]
    in_path = arguments[3]
    out_path = arguments[4]

    if "-e" in key_words.keys():
        extension = key_words["-e"]
    else:
        extension = ""

    if "-i" in key_words.keys():
        in_labels_path = key_words["-i"]
    else:
        in_labels_path = None

    if "-o" in key_words.keys():
        out_labels_path = key_words["-o"]
    else:
        out_labels_path = None

    if "-n" in key_words.keys():
        name = key_words["-n"]
    else:
        name = cfg.retrieve_most_recent_name(config)

    if name is None and cfg.requires_name(config):
        raise exc.AppException.missing_name()

    if name is not None:
        cfg.adjust_paths(config, name, test_paths = False, training_paths = False)

    cfg.ensure_present(out_path)
    if not os.path.exists(config[cfg.CHECKPOINT_PATH]):
        raise( FileNotFoundError("Path not found: %s" % config[cfg.CHECKPOINT_PATH]) )

    print()
    print("Configuration description:")
    print(config[cfg.DESCRIPTION] if len(config[cfg.DESCRIPTION]) > 0 else "None provided")
    print()

    print("Building graph...")

    tf.reset_default_graph()

    if direction.lower() == XY_DIRECTION:
        mode = arch.Mode.RUN_XY
    elif direction.lower() == YX_DIRECTION:
        mode = arch.Mode.RUN_YX
    else:
        raise exc.AppException.invalid_direction()

    if config[cfg.INFO_GAN] or config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        image_size = config[cfg.IMAGE_SIZE]
    else:
        image_size = None

    dataset, _ = input.create_filename_dataset(in_path, shuffle = False, labels_path = in_labels_path, extension = extension)
    dataset = input.create_image_dataset(
        dataset, BATCH_SIZE, EPOCH_COUNT,
        config[cfg.CHANNEL_COUNT_X] if mode == arch.Mode.RUN_XY else config[cfg.CHANNEL_COUNT_Y],
        class_count = config[cfg.CLASS_COUNT], image_size = image_size
    )

    input_item = make_inputs(config, dataset)

    MappingClass, DiscriminatorClass = make_classes(config)

    cycle_gan = arch.CycleGAN(MappingClass, DiscriminatorClass, config, mode = mode)

    if in_labels_path is None or config[cfg.INFO_GAN] or config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        #If CycleGAN uses class information, pass it in alongside the image.
        cycle_gan.create_cycle_gan(input_item.all)
    else:
        #If class information is loaded purely for passing it through, CycleGAN only receives input image.
        cycle_gan.create_cycle_gan(input_item.main)

    if mode == arch.Mode.RUN_XY:
        mapping_output = cycle_gan.mapping_xy.from_input
    else:
        mapping_output = cycle_gan.mapping_yx.from_input

    output_node, file_name = output.output_translated_image(mapping_output.main, out_path, "run_" + direction)

    if out_labels_path is not None:
        label = ops.inverse_one_hot(input_item[1])
        labels = dict()

    print("Initializing session...")

    #Modify GPU memory handling to avoid OOM errors.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  #pylint: disable = E1101

    with tf.Session(config = session_config) as session:
        print("Running...")

        session.run(tf.global_variables_initializer())

        print("Restoring checkpoint from: %s" % config[cfg.CHECKPOINT_PATH])
        checkpoint = tf.train.get_checkpoint_state(config[cfg.CHECKPOINT_PATH])

        if checkpoint and checkpoint.model_checkpoint_path:             #pylint: disable = E1101
            saver = tf.train.Saver(cycle_gan.get_mapping_variables())
            saver.restore(session, checkpoint.model_checkpoint_path)    #pylint: disable = E1101
        else:
            print("Unable to restore checkpoint.")
            return

        while True:
            try:
                if out_labels_path is None:
                    session.run([output_node])
                else:
                    _, file_name_result, label_result = session.run([output_node, file_name, label])
                    labels[file_name_result.decode()] = int(label_result[0])
            except tf.errors.OutOfRangeError:
                break

        if out_labels_path is not None:
            with open(out_labels_path, 'w') as labels_file: json.dump(labels, labels_file)


    print("All done.")


if __name__ == '__main__':
    tf.app.run()