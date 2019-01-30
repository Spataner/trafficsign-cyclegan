'''
Application for clearing directories.
Usage: clear_directory directory_path [files_only] [extension]

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import shutil
import os
import sys


def clear_directory(path, files_only = True, extension = ""):
    '''
    Clears the given directory of files (with a certain extension) and optionally subdirectories.

    Arguments:
        path        -- The path to the directory to clear.
        files_only  -- Whether to only clear files and leave subdirectories untouched.
        extension   -- The file type extension of the files to clear.
    '''

    if not os.path.isdir(path):
        raise Exception("Invalid input path.")

    file_names = [file_name for file_name in os.listdir(path) if (os.path.isfile(os.path.join(path, file_name)) and file_name.endswith(extension))]
    dir_names = [dir_name for dir_name in os.listdir(path) if os.path.isdir(os.path.join(path, dir_name))] if not files_only else list()

    for file_name in file_names:
        os.remove(os.path.join(path, file_name))

    for dir_name in dir_names:
        shutil.rmtree(os.path.join(path, dir_name))

def main(argv):
    argv.pop(0)

    if len(argv) == 0:
        print("Directory path:\t", end = '')
        argv.append(input())

        print("Files only:\t\t", end = '')
        argv.append(input())

        print("Extension:\t\t", end = '')
        argv.append(input())

    if len(argv) > 0 and len(argv) < 4:
        clear_directory(argv[1], argv[2].lower() not in ('false', '0', 'f', 'no', 'n') if len(argv) > 2 else True, argv[3] if len(argv) > 3 else "")
    else:
        print("Usage: clear_directory directory_path [files_only] [extension]")

    


if __name__ == '__main__':
    main(sys.argv)
