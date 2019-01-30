'''
Application for training a multi-class CNN classifier.
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
import tensorflow as tf
import classifier.cnn.input as input
import classifier.cnn.model as model
import classifier.common.report as report


class LoggerHook(tf.train.SessionRunHook):
    '''
    Session hook for training logging.
    Outputs current training information to stdout and writes exemplary image results as well as a Tensorboard summary to file.
    '''

    def __init__(self, loss, epoch_batch_count, epoch_count, log_frequency):
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
        
        self.run_args = list(loss)
        self.summary_writer = None

    def set_summary(self, summary, summary_path):
        '''
        Enable Tensorboard summary.

        Arguments:
            summary         -- The summary object to run.
            summary_path    -- The path to the directory where summary results are saved.
        '''

        if not os.path.isdir(summary_path):
            raise Exception()

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
            print("\tLoss:\t\t", results[0])
            print("\tRuntime:\t\t", str(datetime.timedelta(seconds = runtime)))
            print("\tBatches per minute:\t", str(self.step / runtime * 60))
            print("\tTime remaining:\t\t", str(datetime.timedelta(seconds = runtime / percent - runtime)))
            print()

            if self.summary_writer is not None:
                self.summary_writer.add_summary(results[self.summary_index], global_step = self.step)

    def __del__(self):
        if self.summary_writer is not None:
            self.summary_writer.close()


def main(argv):
    print("Building graph...")

    tf.train.get_or_create_global_step()

    augmenter = lambda x: input.augment_dataset(
            x, [-5, 5], [0.9, 1.1], [-5, 5], [48, 48], 3
        )

    batch_size = 32
    epoch_count = 50
    summary = True

    dataset, epoch_image_count = input.create_filename_dataset(argv[1], shuffle = True, extension = argv[3], labels_path = argv[2])
    dataset = input.create_image_dataset(dataset, batch_size, epoch_count, 3, augmenter = augmenter, class_count = 43, image_size = [48, 48])

    image, label = dataset.make_one_shot_iterator().get_next()

    epoch_batch_count = math.ceil(epoch_image_count / batch_size)

    predicted_label = model.make_classifier(image)

    loss = model.loss(label, predicted_label)

    optimizer = model.optimizer(loss)

    logger_hook = LoggerHook(
        [loss], epoch_batch_count,
        epoch_count, 100
    )

    if summary:
        #Tensorboard summary
        tf.summary.scalar("loss", loss)
        summary = tf.summary.merge_all()
        logger_hook.set_summary(summary, argv[4] + "/summary")

    print("Initializing session...")

    #Modify GPU memory handling to avoid OOM errors.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  #pylint: disable = E1101

    #Modify default scaffold so checkpoints are saved with relative paths, allowing them to be moved/copied/renamed.
    saver = tf.train.Saver(save_relative_paths = True)
    scaffold = tf.train.Scaffold(saver = saver)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir = argv[4] + "/checkpoints", hooks = [logger_hook], config = session_config,
        scaffold = scaffold
    ) as session:

        print("Training...")

        while not session.should_stop():
            session.run([optimizer])

    print("All done.")


if __name__ == '__main__':
    tf.app.run()