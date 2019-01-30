'''
Application for flattening a dataset of nested folders.
Usage: flatten_data in_path out_path [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import os
import shutil
import sys


def collect_filenames(in_path, extension = ""):
    '''
    Collects all file names within a nested structure of folders.

    Arguments:
        in_path     -- The path to the directory to scan.
        extension   -- The file type extension of the files to include in the list.

    Returns:
        A list of file names (full paths) found in the folder.
    '''

    file_names = [os.path.join(in_path, file_name) for file_name in os.listdir(in_path) if os.path.isfile(os.path.join(in_path, file_name)) and file_name.endswith(extension)]
    dir_names = [os.path.join(in_path, dir_name) for dir_name in os.listdir(in_path) if os.path.isdir(os.path.join(in_path, dir_name))]

    for dir_name in dir_names:
        file_names += collect_filenames(dir_name)

    return file_names

def flatten(in_path, out_path, extension = ""):
    '''
    Copies all files found in the input directory and its subdirectories to an output directory.

    Arguments:
        in_path     -- The path to the directory whose files to copy.
        out_path    -- The path to the directory where to copy the files.
    '''

    file_names = collect_filenames(in_path)

    for file_name in file_names:
        flattened_name = file_name.replace(in_path, "").replace("/", "_").replace("\\", "_")
        if flattened_name[0] == "_":
            flattened_name = flattened_name[1:]
        shutil.copyfile(file_name, os.path.join(out_path, flattened_name))

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Input path:\t", end = '')
        argv.append(input())

        print("Output path:\t", end = '')
        argv.append(input())

        print("Extension:\t", end = '')
        argv.append(input())

    if len(argv) in (2, 3):
        flatten(*argv)
    else:
        print("Usage: flatten_data in_path out_path [extension]")


if __name__ == '__main__':
    main(sys.argv)