import json
import os
import sys


def main(argv):
    labels = dict()

    for file_name in os.listdir(argv[1]):
        labels[file_name] = int(file_name.split("_")[0])

    with open(argv[1] + "/labels.json", 'w') as file:
        json.dump(labels, file)


if __name__ == '__main__':
    main(sys.argv)