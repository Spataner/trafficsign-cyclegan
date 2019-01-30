'''
CycleGAN configuration IO.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import json
import datetime
import cyclegan.util.exceptions as exc


#String constants
IMAGE_SIZE = "image_size"
TRAINING_CONFIG = "training_config"
CYCLE_CONSISTENCY_FACTOR = "cycle_consistency_factor"
IMAGE_BUFFER_SIZE = "image_buffer_size"
BATCH_SIZE = "batch_size"
DOMAIN_X_PATH = "domain_x_path"
DOMAIN_Y_PATH = "domain_y_path"
TEST_CONFIG = "test_config"
RES_BLOCK_COUNT = "res_block_count"
CHECKPOINT_PATH = "checkpoint_path"
SUMMARY_PATH = "summary_path"
OUTPUT_PATH = "output_path"
BETA1 = "beta1"
BETA2 = "beta2"
LEARNING_RATE = "learning_rate"
EPOCH_COUNT = "epoch_count"
LOG_FREQUENCY = "log_frequency"
CHANNEL_COUNT_X = "channel_count_x"
CHANNEL_COUNT_Y = "channel_count_y"
DEFAULTED_FIELDS = "defaulted_fields"
RES_BLOCK_RELU = "res_block_relu"
INSTANCE_NORM_AFFINE = "instance_norm_affine"
OUTPUT = "output"
SUMMARY = "summary"
AUGMENT = "augment"
AUGMENT_FLIP = "augment_flip"
AUGMENT_SIZE = "augment_size"
AUGMENT_RANDOM_RESIZE = "augment_random_resize"
DESCRIPTION = "description"
LABELS_X_PATH = "labels_x_path"
LABELS_Y_PATH = "labels_y_path"
CLASS_COUNT = "class_count"
CLASS_MODE = "class_mode"
CLASS_LOSS_TYPE = "class_loss_type"
NONE = "none"
CLASS_CONDITIONAL = "class_conditional"
CLASS_LOSS = "class_loss"
EXTENSION_X = "extension_x"
EXTENSION_Y = "extension_y"
INFO_GAN = "info_gan"
INFO_GAN_SIZE = "info_gan_size"
INFO_GAN_LOSS_FACTOR = "info_gan_loss_factor"
DISCRIMINATOR_X_INITIAL_FILTER_COUNT = "discriminator_x_initial_filter_count"
DISCRIMINATOR_Y_INITIAL_FILTER_COUNT = "discriminator_x_initial_filter_count"
MAPPING_XY_INITIAL_FILTER_COUNT = "mapping_xy_initial_filter_count"
MAPPING_YX_INITIAL_FILTER_COUNT = "mapping_yx_initial_filter_count"
TWON = "2n"
NPLUSONE = "n+1"
NAME_PLACEHOLDER = "[name]"
MAKE_NAME_FORMAT = "%Y%m%d%H%M%S%f"
MAKE_NAME_LENGTH = 20
NUMERIC_SET = set("0123456789")
NESTING_DELIMITER = "/"
WARNING_MESSAGE = "WARNING: The following fields of the loaded configuration were missing and defaulted:"


#Config templates

CONFIG_TEMPLATE_BASE = {
    IMAGE_SIZE : [256, 256],
    CHANNEL_COUNT_X : 3,
    CHANNEL_COUNT_Y : 3,
    RES_BLOCK_COUNT : 9,
    CLASS_COUNT : -1,
    CHECKPOINT_PATH : None,
    RES_BLOCK_RELU : False,
    INSTANCE_NORM_AFFINE : False,
    DESCRIPTION : "",
    CLASS_MODE : NONE,
    INFO_GAN : False,
    INFO_GAN_SIZE : 10,
    MAPPING_XY_INITIAL_FILTER_COUNT : 32,
    MAPPING_YX_INITIAL_FILTER_COUNT : 32
}

CONFIG_TEMPLATE_TRAINING_INNER = {
    TRAINING_CONFIG : {
        CYCLE_CONSISTENCY_FACTOR : 10.0,
        IMAGE_BUFFER_SIZE : 50,
        BATCH_SIZE : 1,
        BETA1 : 0.5,
        BETA2 : 0.999,
        EPOCH_COUNT : 200,
        LEARNING_RATE : 0.0002,
        LOG_FREQUENCY : 100,

        DOMAIN_X_PATH : None,
        DOMAIN_Y_PATH : None,
        LABELS_X_PATH : "",
        LABELS_Y_PATH : "",
        EXTENSION_X : "",
        EXTENSION_Y : "",
        OUTPUT_PATH : None,
        SUMMARY_PATH : None,

        OUTPUT : True,
        SUMMARY : True,

        AUGMENT : True,
        AUGMENT_FLIP : True,
        AUGMENT_SIZE : [286, 286],
        AUGMENT_RANDOM_RESIZE : False,

        CLASS_LOSS_TYPE : NPLUSONE,

        INFO_GAN_LOSS_FACTOR : 0.5,

        DISCRIMINATOR_X_INITIAL_FILTER_COUNT : 64,
        DISCRIMINATOR_Y_INITIAL_FILTER_COUNT : 64
    }
}

CONFIG_TEMPLATE_TEST_INNER = {
    TEST_CONFIG : {
        DOMAIN_X_PATH : None,
        DOMAIN_Y_PATH : None,
        LABELS_X_PATH : "",
        LABELS_Y_PATH : "",
        EXTENSION_X : "",
        EXTENSION_Y : "",
        OUTPUT_PATH : None
    }
}

CONFIG_TEMPLATE_TRAINING = {**CONFIG_TEMPLATE_BASE, **CONFIG_TEMPLATE_TRAINING_INNER}
CONFIG_TEMPLATE_TEST = {**CONFIG_TEMPLATE_BASE, **CONFIG_TEMPLATE_TEST_INNER}
CONFIG_TEMPLATE = {**CONFIG_TEMPLATE_TRAINING, **CONFIG_TEMPLATE_TEST}


def warning(config):
    '''
    Prints a warning concerning defaulted fields in the given configuration.

    Arguments:
        config  -- The configuration that the warning concerns.
    '''

    if len(config[DEFAULTED_FIELDS]) == 0:
        return

    print(WARNING_MESSAGE)
    for defaulted_field in config[DEFAULTED_FIELDS]:
        print("\t" + defaulted_field)

    print()

def check_config(config, template, path = ""):
    '''
    Checks a given configuration against a configuration template, ensuring all values are present.
    If a key with an associated value in the template has no associated value in the configuration, the key in the 
    configuration will be assigned the corresponding value from the template. If the associated value from the template
    is "None", indicating no default value is provided, an exception will be raised instead.

    Arguments:
        config      -- The configuration dictionary to check.
        template    -- The template from which to draw structure and default values.
        path        -- For recursive calls on nested dicts. A string containing the keys of the current dict's parent dicts in the nested structure.

    Returns:
        A list of (in case of nested dicts potentially path-like) keys for which a default value had to be assigned.
    '''

    defaulted_fields = []

    for key in template:
        if key not in config:
            if template[key] is None:
                raise exc.ConfigException.missing_required_field(path + key)
            else:
                defaulted_fields.append(path + key)
                config[key] = template[key]
        else:
            if template[key] is not None and type(template[key]) != type(config[key]):
                raise exc.ConfigException.incorrect_type_field(path + key)
            if type(template[key]) == dict:
                defaulted_fields += check_config(config[key], template[key], path = path + key + NESTING_DELIMITER)

    return defaulted_fields

def load_config(file_name, template = dict()):
    '''
    Loads a configuration from a JSON file into a Python dictionary.

    Arguments:
        file_name   -- The path to the JSON file to read.
        template    -- The configuration template dictionary for checking the loaded configuration.

    Returns:
        The loaded configuration.
    '''

    if not os.path.isfile(file_name):
        raise exc.ConfigException.invalid_file_name()

    with open(file_name, 'r') as in_file:
        loaded_config = json.load(in_file)
        defaulted_fields = check_config(loaded_config, template)
        loaded_config[DEFAULTED_FIELDS] = defaulted_fields
        return loaded_config

def save_config(config, file_name):
    '''
    Saves a given configuration dictionary to a JSON file.

    Arguments:
        config      -- The configuration to save.
        file_name   -- The path to the file to write.
    '''

    config_copy = dict(config)
    del config_copy[DEFAULTED_FIELDS]

    with open(file_name, 'w') as out_file:
        json.dump(config_copy, out_file)

def adjust_paths(config, name, training_paths = True, test_paths = True):
    '''
    Adjusts configuration paths by replacing any name placeholders with the given name.

    Arguments:
        config  -- The configuration whose paths to adjust.
        name    -- The name with which to replace any placeholders.

    Returns:
        The adjusted config (same object as the argument).
    '''

    config[CHECKPOINT_PATH] = config[CHECKPOINT_PATH].replace(NAME_PLACEHOLDER, name)

    if training_paths:
        config[TRAINING_CONFIG][DOMAIN_X_PATH] = config[TRAINING_CONFIG][DOMAIN_X_PATH].replace(NAME_PLACEHOLDER, name)
        config[TRAINING_CONFIG][DOMAIN_Y_PATH] = config[TRAINING_CONFIG][DOMAIN_Y_PATH].replace(NAME_PLACEHOLDER, name)
        config[TRAINING_CONFIG][LABELS_X_PATH] = config[TRAINING_CONFIG][LABELS_X_PATH].replace(NAME_PLACEHOLDER, name)
        config[TRAINING_CONFIG][LABELS_Y_PATH] = config[TRAINING_CONFIG][LABELS_Y_PATH].replace(NAME_PLACEHOLDER, name)

        config[TRAINING_CONFIG][OUTPUT_PATH] = config[TRAINING_CONFIG][OUTPUT_PATH].replace(NAME_PLACEHOLDER, name)
        config[TRAINING_CONFIG][SUMMARY_PATH] = config[TRAINING_CONFIG][SUMMARY_PATH].replace(NAME_PLACEHOLDER, name)

    if test_paths:
        config[TEST_CONFIG][DOMAIN_X_PATH] = config[TEST_CONFIG][DOMAIN_X_PATH].replace(NAME_PLACEHOLDER, name)
        config[TEST_CONFIG][DOMAIN_Y_PATH] = config[TEST_CONFIG][DOMAIN_Y_PATH].replace(NAME_PLACEHOLDER, name)
        config[TEST_CONFIG][LABELS_X_PATH] = config[TEST_CONFIG][LABELS_X_PATH].replace(NAME_PLACEHOLDER, name)
        config[TEST_CONFIG][LABELS_Y_PATH] = config[TEST_CONFIG][LABELS_Y_PATH].replace(NAME_PLACEHOLDER, name)

        config[TEST_CONFIG][OUTPUT_PATH] = config[TEST_CONFIG][OUTPUT_PATH].replace(NAME_PLACEHOLDER, name)

    return config

def make_name():
    '''
    Creates a generic name from the current date and time.

    Returns:
        A string object containing a generic name.
    '''

    now = datetime.datetime.now()
    return now.strftime(MAKE_NAME_FORMAT)

def get_paths(config, training_paths = True, test_paths = True):
    '''
    Rerieves all paths within a configuration.

    Arguments:
        config          -- The configuration from which to retrieve paths.
        training_paths  -- Whether to include training paths in the list.
        test_paths      -- Whether to include test paths in the list.

    Returns:
        A list of all paths in the given configuration.
    '''

    paths = [config[CHECKPOINT_PATH]]

    if training_paths:
        paths += [
            config[TRAINING_CONFIG][DOMAIN_X_PATH], config[TRAINING_CONFIG][DOMAIN_Y_PATH],
            config[TRAINING_CONFIG][OUTPUT_PATH], config[TRAINING_CONFIG][SUMMARY_PATH],
            config[TRAINING_CONFIG][LABELS_X_PATH], config[TRAINING_CONFIG][LABELS_Y_PATH]
        ]

    if test_paths:
        paths += [
            config[TEST_CONFIG][DOMAIN_X_PATH], config[TEST_CONFIG][DOMAIN_Y_PATH],
            config[TEST_CONFIG][OUTPUT_PATH], config[TEST_CONFIG][LABELS_X_PATH],
            config[TEST_CONFIG][LABELS_Y_PATH]
        ]

    return paths

def requires_name(config, training_paths = False, test_paths = True):
    '''
    Determines whether the given configuration contains name placeholders.

    Arguments:
        config          -- The configuration whose paths to check.
        training_path   -- Whether to include training paths in the scan.
        test_path       -- Whether to include test paths in the scan.

    Returns:
        True if the any paths in the configuration contains name placeholder, false otherwise.
    '''

    for path in get_paths(config, training_paths = training_paths, test_paths = test_paths):
        if NAME_PLACEHOLDER in path:
            return True

    return False

def retrieve_names(config, training_paths = False, test_paths = False):
    '''
    Retrieves a list of names for models from the paths in the given config.

    Arguments:
        config          -- The configuration whose paths to check.
        training_path   -- Whether to include training paths in the scan.
        test_path       -- Whether to include test paths in the scan.

    Returns:
        The list of model names.
    '''

    paths = get_paths(config, training_paths = training_paths, test_paths = test_paths)

    paths = [path for path in paths if NAME_PLACEHOLDER in path]
    paths = [path[:path.index(NAME_PLACEHOLDER)] for path in paths]

    if len(paths) == 0:
        return list()

    names = set(os.listdir(paths[0]))
    paths.pop(0)

    for path in paths:
        names &= set(os.listdir(path))

    return list(names)

def retrieve_most_recent_name(config):
    '''
    Retrieves the most recent make_name-style name in use by paths in the config.

    Arguments:
        config -- The config whose paths to check.

    Returns:
        The name of the most recent model.
    '''

    names = retrieve_names(config)
    names = [name for name in names if is_made_name(name)]

    names.sort()

    return names[-1] if len(names) > 0 else None

def is_made_name(name):
    '''
    Determines whether the given name is one potentially created by make_name.

    Arguments:
        name    -- The name to check.

    Returns:
        True if the name could have been created by make_name, False otherwise.
    '''

    return len(name) == MAKE_NAME_LENGTH and is_numeric(name)

def is_numeric(name):
    '''
    Determines whether the given name is purely numeric.

    Arguments:
        name    -- The name to check.

    Returns:
        True if the name contains only numbers, False otherwise.
    '''

    return len(set(name) - NUMERIC_SET) == 0

def ensure_present(path):
    '''
    Creates a directory if it is not already present.

    Arguments:
        path -- The path to the directory whose presence should be ensured.
    '''

    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except:
            raise exc.ConfigException.invalid_output_path()