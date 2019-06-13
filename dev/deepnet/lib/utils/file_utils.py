# module containing methods for file handling
import gzip
import os
import sys
from zipfile import ZipFile


def filehandle_for(filename):
    if filename == "-":
        filehandle = sys.stdin
    else:
        filehandle = open(filename)
    return filehandle


def list_files_in_dir(path, extension='*'):
    file_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if extension in file:
                file_paths.append(os.path.join(root, file))
    return file_paths


def write(path, content):
    file = open(path, 'w')
    file.write(content)
    file.close()


def unzip_if_zipped(zipped_file):
    if ".gz" in zipped_file:
        file = gzip.open(zipped_file, 'rb')
    elif ".zip" in zipped_file:
        file = ZipFile(zipped_file).extractall()
    else:
        file = zipped_file
    return file
