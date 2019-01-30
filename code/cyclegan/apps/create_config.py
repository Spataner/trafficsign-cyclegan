'''
Guided creation of CycleGAN configuration files.
Usage: create_config out_config_path

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sys
import cyclegan.util.config as cfg


CONFIG_INFO_BASE = {
    cfg.IMAGE_SIZE : "A list of two integers specifying size (height, width) of the images in both domains.",
    cfg.CHANNEL_COUNT_X : "The number of channels of images in domain X (usually either 1 or 3).",
    cfg.CHANNEL_COUNT_Y : "The number of channels of images in domain Y (usually either 1 or 3).",
    cfg.RES_BLOCK_COUNT : "The number of residual blocks used in the generators.",
    cfg.CHECKPOINT_PATH : "The path to the directory where training checkpoints will be saved to and loaded from.",
    cfg.RES_BLOCK_RELU : "Whether the generators' residual blocks shall use ReLU activation on their output.",
    cfg.INSTANCE_NORM_AFFINE : "Whether the instance norm layers shall include a trainable affine transform.",
    cfg.DESCRIPTION : "A description of the configuration.",
    cfg.CLASS_MODE : "The manner in which class information is used. May be \"none\", \"class_conditional\", or \"class_loss\".",
    cfg.CLASS_COUNT : "The number of classes present in the data. Only used when \"class_mode\" is not \"none\".",
    cfg.INFO_GAN : "Whether to use the infoGAN extension. This field is ignored if \"class_mode\" is not \"none\".",
    cfg.INFO_GAN_SIZE : "The size of the infoGAN latent variable vector.",
    cfg.MAPPING_XY_INITIAL_FILTER_COUNT : "The number of filters for the first layer of the XY mapping (subsequent layers use multiplies of this value).",
    cfg.MAPPING_YX_INITIAL_FILTER_COUNT : "The number of filters for the first layer of the YX mapping (subsequent layers use multiplies of this value)."
}

CONFIG_INFO_TRAINING_INNER = {
    cfg.TRAINING_CONFIG : {
        cfg.CYCLE_CONSISTENCY_FACTOR : "The multiplicative factor of the cycle-consistency loss in the mappings' loss calculation.",
        cfg.IMAGE_BUFFER_SIZE : "The number of generated images buffered for the purposes of discriminator training.",
        cfg.BATCH_SIZE : "The size of the training batches (usually 1).",
        cfg.BETA1 : "The beta1 parameter for the Adam optimizer.",
        cfg.BETA2 : "The beta2 parameter for the Adam optimizer.",
        cfg.EPOCH_COUNT : "The number of epochs for which to train.",
        cfg.LEARNING_RATE : "The initial (undecayed) learning rate.",
        cfg.LOG_FREQUENCY : "The frequency (in steps) at which training logging occurs.",

        cfg.DOMAIN_X_PATH : "The path to the directory that holds the training input images for domain X.",
        cfg.DOMAIN_Y_PATH : "The path to the directory that holds the training input images for domain Y.",
        cfg.LABELS_X_PATH : "The path to the JSON file that holds the labels for the training input images of domain X",
        cfg.LABELS_Y_PATH : "The path to the JSON file that holds the labels for the training input images of domain Y",
        cfg.OUTPUT_PATH : "The path to the image output directory for training.",
        cfg.SUMMARY_PATH : "The path to the directory where the Tensorboard summary shall be saved.",
        cfg.EXTENSION_X : "The file type extension of images to load for domain X.",
        cfg.EXTENSION_Y : "The file type extension of images to load for domain Y.",

        cfg.SUMMARY : "Whether to output a Tensorboard summary.",
        cfg.OUTPUT : "Whether to output training images.",

        cfg.AUGMENT : "Whether to use runtime dataset augmentation.",
        cfg.AUGMENT_FLIP : "Whether to randomly horizontally flip images during data augmentation.",
        cfg.AUGMENT_SIZE : "The size images are upscaled to before randomly recropping to their original size during augmentation.",
        cfg.AUGMENT_RANDOM_RESIZE : "Whether to treat \"augment_size\" as the upper bound for a upscale size determined uniformly at random.",

        cfg.CLASS_LOSS_TYPE : "The manner in which class based discriminator loss is defined. May be either \"n+1\" or \"2n\". Only used when \"class_mode\" is \"class_loss\".",
        cfg.INFO_GAN_LOSS_FACTOR : "The factor that weights the infoGAN loss relative to the regular GAN loss.",

        cfg.DISCRIMINATOR_X_INITIAL_FILTER_COUNT : "The number of filters for the first layer of the X discriminator (subsequent layers use multiplies of this value).",
        cfg.DISCRIMINATOR_Y_INITIAL_FILTER_COUNT : "The number of filters for the first layer of the Y discriminator (subsequent layers use multiplies of this value)."
    }
}

CONFIG_INFO_TEST_INNER = {
    cfg.TEST_CONFIG : {
        cfg.DOMAIN_X_PATH : "The path to the directory that holds the test input images for domain X.",
        cfg.DOMAIN_Y_PATH : "The path to the directory that holds the test input images for domain Y.",
        cfg.LABELS_X_PATH : "The path to the JSON file that holds the labels for the test input images of domain X",
        cfg.LABELS_Y_PATH : "The path to the JSON file that holds the labels for the test input images of domain Y",
        cfg.OUTPUT_PATH : "The path to the directory to which test results shall be saved.",
        cfg.EXTENSION_X : "The file type extension of images to load for domain X.",
        cfg.EXTENSION_Y : "The file type extension of images to load for domain Y."
    }
}

CONFIG_INFO_TRAINING = {**CONFIG_INFO_BASE, **CONFIG_INFO_TRAINING_INNER}
CONFIG_INFO_TEST = {**CONFIG_INFO_BASE, **CONFIG_INFO_TEST_INNER}
CONFIG_INFO = {**CONFIG_INFO_TRAINING, **CONFIG_INFO_TEST}


def get_value(config, key, index):
    '''
    Retrieves a value from the config or a list within the dict.
    
    Arguments:
        config      -- The config from which to retrieve the value.
        key         -- The key for the value within the config.
        index       -- The index of the desired value if the value at "key" is a list. May be None.

    Returns:
        The retrieved value.
    '''

    if index is None:
        return config[key]
    return config[key][index]

def set_value(config, key, value, index):
    '''
    Sets a value in the config or a list within the dict.
    
    Arguments:
        config      -- The config where to set the value.
        key         -- The key for the value within the config.
        index       -- The index of the value if the value at "key" is a list. May be None.
    '''

    if index is None:
        config[key] = value
    else:
        config[key][index] = value

def request_field(config, config_info, template, key, depth, index = None):
    '''
    Requests a value input from the user.

    Arguments:
        config      -- The config for which to request the value.
        config_info -- The dict containing further information about the values for the user.
        template    -- The config template for default values.
        key         -- The key of the value to request.
        depth       -- The current depth of recursion (for identation).
        index       -- The index of the value to request if the value at "key" is a list. May be None.
    '''

    while True:
        print(("\t" * depth) + key + ("[" + str(index) + "]" if index is not None else "") + ": ", end = "")
        value = input()

        if value == "":
            if get_value(template, key, index) is None:
                print(("\t" * depth) + "This is a required field and cannot be defaulted.")
            else:
                print(("\t" * depth) + "Using default value (" + str(get_value(template, key, index)) + ").")
                set_value(config, key, get_value(template, key, index), index)
                break
        elif value == "?":
            print(("\t" * depth) + config_info[key])
        else:
            try:
                value = type(get_value(template, key, index))(value)
                set_value(config, key, value, index)
                break
            except ValueError:
                print("Invalid value; must be type " + str(type(get_value(template, key, index))) + ".")
            except TypeError:
                set_value(config, key, value, index)
                break

def request_fields(config, config_info, template, depth = 0):
    '''
    Requests a value input from the user for all fields in a config.

    Arguments:
        config      -- The config for which to request fields.
        config_info -- The dict containing further information about the values for the user.
        template    -- The config template for default values and structure.
        depth       -- The current depth of recursion (for identation).
    '''

    for key in template:
        if type(template[key]) == dict:
            print(("\t" * depth) + key + ":")
            config[key] = dict()
            request_fields(config[key], config_info[key], template[key], depth = depth+1)
        elif type(template[key]) == list:
            config[key] = [None] * len(template[key])
            for i in range(len(template[key])):
                request_field(config, config_info, template, key, depth, index = i)
        else:
            request_field(config, config_info, template, key, depth)

def main(argv):
    if len(argv) < 2:
        print("Config output path: ", end = "")
        argv.append(input())

    try:
        file = open(argv[1], 'w')
    except:
        print("Invalid output path.")
        return
    finally:
        file.close()

    print()
    print("Creating new configuration (type \"?\" for more information regarding the current field).")
    print()

    config = dict()
    request_fields(config, CONFIG_INFO, cfg.CONFIG_TEMPLATE)
    config[cfg.DEFAULTED_FIELDS] = list()

    cfg.save_config(config, argv[1])

    print()
    print("All done.")


if __name__ == '__main__':
    main(sys.argv)