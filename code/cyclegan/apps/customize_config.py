'''
Guided customization of the directory paths within a given configuration file.
Usage: create_config in_config_path out_config_path

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import cyclegan.util.config as cfg


def main(argv):
    if len(argv) == 1:
        print("Input config path:\t", end = '')
        argv.append(input())
        print("Output config path:\t", end = '')
        argv.append(input())

    elif len(argv) == 2:
        print("Usage: customize_config in_config_path out_config_path")
        return

    config = cfg.load_config(argv[1], template = cfg.CONFIG_TEMPLATE)


    print("Customizing configuration.")
    print()


    print("checkpoint_path:\t", end = "")
    config[cfg.CHECKPOINT_PATH] = input()


    print("training_config:")

    print("\tdomain_x_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.DOMAIN_X_PATH] = input()

    print("\tdomain_y_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.DOMAIN_Y_PATH] = input()

    print("\toutput_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.OUTPUT_PATH] = input()

    print("\tsummary_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.SUMMARY_PATH] = input()

    print("\tlabels_x_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.LABELS_X_PATH] = input()

    print("\tlabels_y_path:\t", end = "")
    config[cfg.TRAINING_CONFIG][cfg.LABELS_Y_PATH] = input()


    print ("test_config:")

    print("\tdomain_x_path:\t", end = "")
    config[cfg.TEST_CONFIG][cfg.DOMAIN_X_PATH] = input()

    print("\tdomain_y_path:\t", end = "")
    config[cfg.TEST_CONFIG][cfg.DOMAIN_Y_PATH] = input()

    print("\toutput_path:\t", end = "")
    config[cfg.TEST_CONFIG][cfg.OUTPUT_PATH] = input()

    print("\tlabels_x_path:\t", end = "")
    config[cfg.TEST_CONFIG][cfg.LABELS_X_PATH] = input()

    print("\tlabels_y_path:\t", end = "")
    config[cfg.TEST_CONFIG][cfg.LABELS_Y_PATH] = input()


    cfg.save_config(config, argv[2])


    print()
    print("All done.")


if __name__ == '__main__':
    main(sys.argv)