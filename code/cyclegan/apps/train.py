'''
Application for training a CycleGAN model.
Usage: train in_config_path

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import datetime
import time
import math
import os
import tensorflow                   as tf
import cyclegan.model.train         as train
import cyclegan.util.config         as cfg
import cyclegan.model.input         as input
import cyclegan.model.loss          as loss
import cyclegan.model.architecture  as arch
import cyclegan.util.exceptions     as exc
import cyclegan.model.output        as output


class LoggerHook(tf.train.SessionRunHook):
    '''
    Session hook for training logging.
    Outputs current training information to stdout and writes exemplary image results as well as a Tensorboard summary to file.
    '''

    def __init__(self, losses, epoch_batch_count, epoch_count, log_frequency):
        '''
        Setter constructor.

        Arguments:
            losses              -- List of three scalar tensors representing the losses of discriminator X, discriminator Y, and the mappings.
            epoch_batch_count   -- The number of batches per epoch.
            epoch_count         -- The number of epochs.
            log_frequency       -- The frequency in steps/batches that logging will occur.
        '''

        self.epoch_batch_count = epoch_batch_count
        self.log_frequency = log_frequency
        self.total_steps = epoch_batch_count * epoch_count
        
        self.run_args = list(losses)
        self.summary_writer = None

    def set_summary(self, summary, summary_path):
        '''
        Enable Tensorboard summary.

        Arguments:
            summary         -- The summary object to run.
            summary_path    -- The path to the directory where summary results are saved.
        '''

        if not os.path.isdir(summary_path):
            raise exc.AppException.invalid_summary_path()

        self.summary_writer = tf.summary.FileWriter(summary_path, graph = tf.get_default_graph())
        self.run_args.append(summary)
        self.summary_index = len(self.run_args) - 1

    def set_outputs(self, outputs):
        '''
        Enables training image output.

        Arguments:
            outputs     -- The list of outputs to run.
        '''

        self.run_args += outputs

    def begin(self):
        self.step = -1

    def before_run(self, run_context):
        self.step += 1
        if self.step == 0:
            self.start_time = time.time()
            
        return tf.train.SessionRunArgs(self.run_args if self.do_log() else list())

    def do_log(self):
        '''
        Whether logging shall occur in the current step.

        Returns:
            True if logging shall occur in this step; False otherwise.
        '''

        return self.step % self.log_frequency == 0 and self.step != 0

    def after_run(self, run_context, run_values):
        if self.do_log():
            results = run_values.results
            percent = (self.step+1) / self.total_steps
            runtime = time.time() - self.start_time

            print("Step " + str(self.step) + ", epoch " + str(self.step // self.epoch_batch_count) + " (" + str(int(100 * percent)) + "% complete):")
            print("\tMapping loss:\t\t", results[0])
            print("\tDiscriminator losses:\t", results[1], results[2])
            print("\tRuntime:\t\t", str(datetime.timedelta(seconds = runtime)))
            print("\tBatches per minute:\t", str(self.step / runtime * 60))
            print("\tTime remaining:\t\t", str(datetime.timedelta(seconds = runtime / percent - runtime)))
            print()

            if self.summary_writer is not None:
                self.summary_writer.add_summary(results[self.summary_index], global_step = self.step)

    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.close()


def make_datasets(config):
    '''
    Creates a pair of datasets from a config.

    Arguments:
        config -- The configuration to use for the creation of the datasets.

    Returns:
        One dataset for each domain as well as the number of images per epoch.
    '''

    if config[cfg.TRAINING_CONFIG][cfg.AUGMENT]:
        augmenter = lambda x, y: input.augment_dataset(
            x, config[cfg.TRAINING_CONFIG][cfg.AUGMENT_SIZE], config[cfg.IMAGE_SIZE],
            y, flip = config[cfg.TRAINING_CONFIG][cfg.AUGMENT_FLIP],
            random_resize = config[cfg.TRAINING_CONFIG][cfg.AUGMENT_RANDOM_RESIZE]
        )
    else:
        augmenter = None

    if config[cfg.CLASS_MODE] == cfg.NONE:
        dataset_x, dataset_y, epoch_image_count = input.create_domain_datasets(
            config[cfg.TRAINING_CONFIG][cfg.DOMAIN_X_PATH], config[cfg.TRAINING_CONFIG][cfg.DOMAIN_Y_PATH],
            config[cfg.TRAINING_CONFIG][cfg.BATCH_SIZE], config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT],
            config[cfg.IMAGE_SIZE], config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y], augmenter = augmenter,
            extension_x = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_X],
            extension_y = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_Y]
        )
    elif config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        dataset_x, dataset_y, epoch_image_count = input.create_domain_datasets(
            config[cfg.TRAINING_CONFIG][cfg.DOMAIN_X_PATH], config[cfg.TRAINING_CONFIG][cfg.DOMAIN_Y_PATH],
            config[cfg.TRAINING_CONFIG][cfg.BATCH_SIZE], config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT],
            config[cfg.IMAGE_SIZE], config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y], augmenter = augmenter,
            labels_x_path = config[cfg.TRAINING_CONFIG][cfg.LABELS_X_PATH],
            labels_y_path = config[cfg.TRAINING_CONFIG][cfg.LABELS_Y_PATH],
            class_count = config[cfg.CLASS_COUNT],
            extension_x = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_X],
            extension_y = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_Y]
        )
    else:
        if config[cfg.TRAINING_CONFIG][cfg.CLASS_LOSS_TYPE] == cfg.NPLUSONE:
            class_count = config[cfg.CLASS_COUNT] + 1
        else:
            class_count = 2 * config[cfg.CLASS_COUNT]

        dataset_x, dataset_y, epoch_image_count = input.create_domain_datasets(
            config[cfg.TRAINING_CONFIG][cfg.DOMAIN_X_PATH], config[cfg.TRAINING_CONFIG][cfg.DOMAIN_Y_PATH],
            config[cfg.TRAINING_CONFIG][cfg.BATCH_SIZE], config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT],
            config[cfg.IMAGE_SIZE], config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y], augmenter = augmenter,
            labels_x_path = config[cfg.TRAINING_CONFIG][cfg.LABELS_X_PATH],
            labels_y_path = config[cfg.TRAINING_CONFIG][cfg.LABELS_Y_PATH],
            class_count = class_count,
            extension_x = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_X],
            extension_y = config[cfg.TRAINING_CONFIG][cfg.EXTENSION_Y]
        )

    return dataset_x, dataset_y, epoch_image_count

def make_inputs(config, dataset):
    '''
    Creates a wrapped input tensor tuple from a configurationa and dataset.

    Arguments:
        config      -- The configuration on which to base the tensor tuple.
        dataset     -- The dataset from which to create the tensor tuple.

    Returns:
        The wrapped input tensor tuple.
    '''

    next = dataset.make_one_shot_iterator().get_next()

    if config[cfg.INFO_GAN]:
        if type(next) != tuple:
            next = (next,)

        info_gan_vector = tf.random_uniform([tf.shape(next[0])[0], config[cfg.INFO_GAN_SIZE]], dtype = tf.float32) * 2 - 1

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

    if config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        MappingClass = arch.get_vector_decorated_mapping([True], [True])
        DiscriminatorClass = arch.get_vector_decorated_discriminator([True], [True])
    elif config[cfg.CLASS_MODE] == cfg.CLASS_LOSS:
        MappingClass = arch.get_vector_decorated_mapping([False], [True])
        DiscriminatorClass = arch.get_vector_decorated_discriminator([False], [True])
    elif config[cfg.INFO_GAN]:
        MappingClass = arch.get_vector_decorated_mapping([True], [True])
        DiscriminatorClass = arch.get_info_gan_discriminator(config[cfg.INFO_GAN_SIZE])
    else:
        MappingClass = arch.Mapping
        DiscriminatorClass = arch.Discriminator

    return MappingClass, DiscriminatorClass

def make_learning_rate(config, epoch_batch_count):
    '''
    Creates the learning rate from a config.

    Arguments:
        config              -- The configuration on which to base the learning rate.
        epoch_batch_count   -- The number of batches per epoch.

    Returns:
        The learning rate tensor.
    '''

    constant_epoch_count = config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT] // 2
    linear_epoch_count = config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT] - constant_epoch_count
    return train.create_delayed_linear_decay(config[cfg.TRAINING_CONFIG][cfg.LEARNING_RATE], constant_epoch_count, linear_epoch_count, epoch_batch_count)

def make_losses(config, input_x, input_y, cycle_gan):
    '''
    Creates the training losses according to the given config.

    Arguments:
        config      -- The configuration on which to base the losses.
        input_x     -- The (wrapped) input for domain X.
        input_y     -- The (wrapped) input for domain Y.
        cycle_gan   -- The CycleGAN object.

    Returns:
        Three loss tensors for mapping, X discriminator, and Y discriminator.
    '''

    if config[cfg.CLASS_MODE] == cfg.CLASS_LOSS:
        if config[cfg.TRAINING_CONFIG][cfg.CLASS_LOSS_TYPE] == cfg.NPLUSONE:
            mapping_loss = loss.mapping_loss_class(
                input_x.main, input_y.main, cycle_gan, config[cfg.TRAINING_CONFIG][cfg.CYCLE_CONSISTENCY_FACTOR],
                config[cfg.CLASS_COUNT] + 1
            )
            discriminator_x_loss = loss.discriminator_loss_nplusone_class(
                cycle_gan.discriminator_x, config[cfg.CLASS_COUNT] + 1
            )
            discriminator_y_loss = loss.discriminator_loss_nplusone_class(
                cycle_gan.discriminator_y, config[cfg.CLASS_COUNT] + 1
            )
        else:
            mapping_loss = loss.mapping_loss_class(
                input_x.main, input_y.main, cycle_gan, config[cfg.TRAINING_CONFIG][cfg.CYCLE_CONSISTENCY_FACTOR],
                config[cfg.CLASS_COUNT] * 2
            )
            discriminator_x_loss = loss.discriminator_loss_twon_class(
                cycle_gan.discriminator_x, config[cfg.CLASS_COUNT] * 2
            )
            discriminator_y_loss = loss.discriminator_loss_twon_class(
                cycle_gan.discriminator_y, config[cfg.CLASS_COUNT] * 2
            )
    elif config[cfg.CLASS_MODE] == cfg.CLASS_CONDITIONAL:
        mapping_loss = loss.mapping_loss(input_x.main, input_y.main, cycle_gan, config[cfg.TRAINING_CONFIG][cfg.CYCLE_CONSISTENCY_FACTOR])
        discriminator_x_loss = loss.discriminator_loss(cycle_gan.discriminator_x)
        discriminator_y_loss = loss.discriminator_loss(cycle_gan.discriminator_y)
    elif config[cfg.INFO_GAN]:
        mapping_loss = loss.mapping_loss(input_x.main, input_y.main, cycle_gan, config[cfg.TRAINING_CONFIG][cfg.CYCLE_CONSISTENCY_FACTOR])

        discriminator_x_loss = loss.discriminator_loss(cycle_gan.discriminator_x) 
        discriminator_x_loss += config[cfg.TRAINING_CONFIG][cfg.INFO_GAN_LOSS_FACTOR] * loss.info_gan_loss(cycle_gan.discriminator_x)

        discriminator_y_loss = loss.discriminator_loss(cycle_gan.discriminator_y)
        discriminator_y_loss += config[cfg.TRAINING_CONFIG][cfg.INFO_GAN_LOSS_FACTOR] * loss.info_gan_loss(cycle_gan.discriminator_y)
    else:
        mapping_loss = loss.mapping_loss(input_x.main, input_y.main, cycle_gan, config[cfg.TRAINING_CONFIG][cfg.CYCLE_CONSISTENCY_FACTOR])
        discriminator_x_loss = loss.discriminator_loss(cycle_gan.discriminator_x)
        discriminator_y_loss = loss.discriminator_loss(cycle_gan.discriminator_y)

    return mapping_loss, discriminator_x_loss, discriminator_y_loss

def main(argv):
    if len(argv) < 2:
        raise exc.AppException.missing_config()

    print("Started CycleGAN training.")
    print()

    print("Loading configuration...")

    config = cfg.load_config(argv[1], template = cfg.CONFIG_TEMPLATE_TRAINING)

    cfg.warning(config)

    if len(argv) > 2:
        name = argv[2]
    else:
        name = cfg.make_name()

    cfg.adjust_paths(config, name, test_paths = False)
    cfg.ensure_present(config[cfg.CHECKPOINT_PATH])

    print()
    print("Configuration description:")
    print(config[cfg.DESCRIPTION] if len(config[cfg.DESCRIPTION]) > 0 else "None provided")
    print()

    print("Building graph...")

    tf.train.get_or_create_global_step()

    dataset_x, dataset_y, epoch_image_count = make_datasets(config)
    input_x = make_inputs(config, dataset_x)
    input_y = make_inputs(config, dataset_y)

    MappingClass, DiscriminatorClass = make_classes(config)

    epoch_batch_count = math.ceil(epoch_image_count / config[cfg.TRAINING_CONFIG][cfg.BATCH_SIZE])
    learning_rate = make_learning_rate(config, epoch_batch_count)

    cycle_gan = arch.CycleGAN(MappingClass, DiscriminatorClass, config)
    cycle_gan.create_cycle_gan(input_x.all, input_y.all)

    mapping_loss, discriminator_x_loss, discriminator_y_loss = make_losses(config, input_x, input_y, cycle_gan)

    mapping_training_op, discriminator_x_training_op, discriminator_y_training_op = train.create_training_ops(
        cycle_gan, mapping_loss, discriminator_x_loss, discriminator_y_loss,
        learning_rate, config[cfg.TRAINING_CONFIG][cfg.BETA1],
        config[cfg.TRAINING_CONFIG][cfg.BETA2]
    )

    logger_hook = LoggerHook(
        [mapping_loss, discriminator_x_loss, discriminator_y_loss], epoch_batch_count,
        config[cfg.TRAINING_CONFIG][cfg.EPOCH_COUNT], config[cfg.TRAINING_CONFIG][cfg.LOG_FREQUENCY]
    )

    maximum_channel_count = max([config[cfg.CHANNEL_COUNT_X], config[cfg.CHANNEL_COUNT_Y]])

    if config[cfg.TRAINING_CONFIG][cfg.OUTPUT]:
        cfg.ensure_present(config[cfg.TRAINING_CONFIG][cfg.OUTPUT_PATH])

        output_x = output.output_test_image(
            input_x.main, cycle_gan.mapping_xy.from_input.main, cycle_gan.mapping_yx.cycle.main, config[cfg.TRAINING_CONFIG][cfg.OUTPUT_PATH],
            "training_xyx", target_channel_count = maximum_channel_count
        )
        output_y = output.output_test_image(
            input_y.main, cycle_gan.mapping_yx.from_input.main, cycle_gan.mapping_xy.cycle.main, config[cfg.TRAINING_CONFIG][cfg.OUTPUT_PATH],
            "training_yxy", target_channel_count = maximum_channel_count
        )
        
        logger_hook.set_outputs([output_x, output_y])

    if config[cfg.TRAINING_CONFIG][cfg.SUMMARY]:
        cfg.ensure_present(config[cfg.TRAINING_CONFIG][cfg.SUMMARY_PATH])

        #Tensorboard summary
        tf.summary.scalar("discriminator_x_loss", discriminator_x_loss)
        tf.summary.scalar("discriminator_y_loss", discriminator_y_loss)
        tf.summary.scalar("mapping_loss", mapping_loss)
        tf.summary.image("training_xyx", output.concatenate_images(
            output.convert_image(input_x.main), output.convert_image(cycle_gan.mapping_xy.from_input.main),
            output.convert_image(cycle_gan.mapping_yx.cycle.main)
        ))
        tf.summary.image("training_yxy", output.concatenate_images(
            output.convert_image(input_y.main), output.convert_image(cycle_gan.mapping_yx.from_input.main),
            output.convert_image(cycle_gan.mapping_xy.cycle.main)
        ))
        summary = tf.summary.merge_all()
        logger_hook.set_summary(summary, config[cfg.TRAINING_CONFIG][cfg.SUMMARY_PATH])

    print()
    print("Parameter counts:")
    print("\tDiscriminators:\t", cycle_gan.discriminator_x.parameter_count(), cycle_gan.discriminator_y.parameter_count())
    print("\tMappings:\t", cycle_gan.mapping_xy.parameter_count(), cycle_gan.mapping_yx.parameter_count())
    print()

    print("Initializing session...")

    #Modify GPU memory handling to avoid OOM errors.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  #pylint: disable = E1101

    #Modify default scaffold so checkpoints are saved with relative paths, allowing them to be moved/copied/renamed.
    saver = tf.train.Saver(save_relative_paths = True)
    scaffold = tf.train.Scaffold(saver = saver)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = config[cfg.CHECKPOINT_PATH], hooks = [logger_hook], config = session_config,
        scaffold = scaffold
    ) as session:

        print("Training...")

        while not session.should_stop():
            session.run([mapping_training_op, discriminator_x_training_op, discriminator_y_training_op])

    print("All done.")


if __name__ == '__main__':
    tf.app.run()