# module containing methods for file handling
import logging
import os
import subprocess
from zipfile import ZipFile

from .exceptions import ProcessError

logger = logging.getLogger('root')


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
    if '.gz' in zipped_file or '.zip' in zipped_file:
        try:
            if ".gz" in zipped_file:
                # subprocess.run(['gzip', '-d', zipped_file], check=True)
                subprocess.run(['gzip', zipped_file], check=True)
                file = zipped_file.replace('.gz', '')
            elif ".zip" in zipped_file:
                with ZipFile(zipped_file, 'r') as zipObj:
                    zipObj.extractall(os.path.dirname(zipped_file))
                file = zipped_file.replace('.zip', '')
                # if os.path.isfile(file):
                #     os.remove(zipped_file)
        except Exception as e:
            logger.error(e)
            raise ProcessError(F'Failed to unzip file {zipped_file}. Please provide the files in an uncompressed format.')
    else:
        file = zipped_file
    return file
