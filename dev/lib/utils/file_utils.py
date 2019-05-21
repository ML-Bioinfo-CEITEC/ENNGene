# module containing methods for file handling
import os
import sys


def filehandle_for(filename):
    if filename == "-":
        filehandle = sys.stdin
    else:
        filehandle = open(filename)
    return filehandle


def list_files_in_dir(path, extension='*'):
    files = []
    for root, _, files in os.walk(path):
        for file in files:
            if extension in file:
                files.append(os.path.join(root, file))
    return files


def dictionary_to_bed(dictionary, path):
    return


def dictionary_to_bedgraph(dataset, path):
    return
