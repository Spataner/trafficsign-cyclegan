'''
CycleGAN exception classes.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


class ConfigException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

    @staticmethod
    def invalid_file_name():
        return ConfigException("Invalid file name.")

    @staticmethod
    def missing_required_field(path):
        return ConfigException("Config is missing required field \"" + path + "\".")

    @staticmethod
    def incorrect_type_field(path):
        return ConfigException("Config's field \"" + path + "\" has incorrect type.")

    @staticmethod
    def invalid_output_path():
        return AppException("The fiven output path is not a valid path for a directory.")


class AppException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

    @staticmethod
    def missing_config():
        return AppException("No config provided.")

    @staticmethod
    def missing_arguments():
        return AppException("Application missing arguments (required: config_path direction in_path out_path).")

    @staticmethod
    def invalid_direction():
        return AppException("The given direction is invalid (must be either \"xy\" or \"yx\").")

    @staticmethod
    def missing_name():
        return AppException("A model name is required, yet was neither provided nor could be inferred.")

    @staticmethod
    def invalid_summary_path():
        return AppException("The summary path is not a directory.")


class InputException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

    @staticmethod
    def invalid_input_path():
        return InputException("Input path is not a directory.")

    @staticmethod
    def invalid_labels_path():
        return InputException("Labels path is not a file.")

    @staticmethod
    def empty_directory():
        return InputException("The input directory is empty.")


class OutputException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

    @staticmethod
    def invalid_path():
        return OutputException("Output path is not a directory.")


class ArchitectureException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

    @staticmethod
    def incorrect_indexed_access():
        return ArchitectureException("Attempted indexed access on non-tuple output.")