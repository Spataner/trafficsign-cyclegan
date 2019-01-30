'''
Application for testing a mutli-class CNN classifier.
Usage: test in_config_path

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import numpy as np
import tensorflow as tf
import classifier.cnn.model as model
import classifier.cnn.input as input
import classifier.common.report as report


EPOCH_COUNT = 1
BATCH_SIZE = 1


def main(argv):
    print("Building graph...")

    tf.reset_default_graph()

    dataset, _ = input.create_filename_dataset(argv[1], shuffle = False, extension = argv[3], labels_path = argv[2])
    dataset = input.create_image_dataset(dataset, BATCH_SIZE, EPOCH_COUNT, 3, augmenter = None, image_size = [48, 48], class_count = 43)

    image, label = dataset.make_one_shot_iterator().get_next()

    predicted_label = model.make_classifier(image)

    numeric_label = model.inverse_one_hot(label)
    numeric_predicted_label = tf.argmax(predicted_label, 1)

    print("Initializing session...")

    #Modify GPU memory handling to avoid OOM errors.
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True  #pylint: disable = E1101

    labels = list()
    predicted_labels = list()

    with tf.Session(config = session_config) as session:
        print("Testing...")

        session.run(tf.global_variables_initializer())

        checkpoint = tf.train.get_checkpoint_state(argv[4] + "/checkpoints")

        if checkpoint and checkpoint.model_checkpoint_path:             #pylint: disable = E1101
            saver = tf.train.Saver()
            saver.restore(session, checkpoint.model_checkpoint_path)    #pylint: disable = E1101
        else:
            print("Unable to restore checkpoint.")
            return

        while True:
            try:
                label_val, predicted_label_val = session.run([numeric_label, numeric_predicted_label])
                labels.append(label_val[0])
                predicted_labels.append(predicted_label_val[0])
            except tf.errors.OutOfRangeError:
                break

    result = report.Report(np.array(labels), np.array(predicted_labels), report.LABEL_NAMES)

    print("Error: ", result.error)
    print()
    print(result.get_sparse_string(result.confusion_matrix_relative, 0.01))

    with open(argv[4] + "/report.tex", 'w') as report_file: result.dump(report_file)

    print("All done.")


if __name__ == '__main__':
    tf.app.run()