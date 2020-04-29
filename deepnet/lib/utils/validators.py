import os

from .dataset import Dataset
from . import file_utils as f


def not_empty_branches(branches):
    warning = 'You must select at least one branch.'
    return warning if len(branches) == 0 else None


def min_one_file(input_files):
    warning = 'You must provide at least one input file.'
    return warning if len(input_files) == 0 else None


def is_bed(file):
    invalid = False
    if os.path.isfile(file):
        with open(file) as f:
            try:
                line = f.readline()
            except Exception:
                invalid = True
            cells = line.split('\t')
            if len(cells) >= 3:
                try:
                    int(cells[1])
                    int(cells[2])
                except Exception:
                    invalid = True
            else:
                invalid = True
    else:
        invalid = True

    warning = f"File {file} does not look like valid BED file (tab separated)."
    return warning if invalid else None


def is_fasta(file):
    invalid = False
    if os.path.isfile(file):
        with open(file) as f:
            try:
                line1 = f.readline()
                line2 = f.readline()
            except Exception:
                invalid = True
            if not ('>' in line1) or not line2:
                invalid = True

    warning = f"File {file} does not look like valid FASTA file."
    return warning if invalid else None


def is_wig_dir(folder):
    # Checks just one random (first found) wig file

    invalid = False
    if os.path.isdir(folder):
        files = f.list_files_in_dir(folder)
        one_wig = next((file for file in files if 'wig' in file), None)
        if one_wig:
            zipped = True if ('gz' in one_wig or 'zip' in one_wig) else False
            try:
                with open(one_wig) as wig_file:
                    line1 = f.read_decoded_line(wig_file, zipped)
                    if not ('fixedStep' in line1 or 'variableStep' in line1) or not ('chrom' in line1):
                        invalid = True
                        warning = f"Provided wig file {one_wig} starts with unknown header."
                    line2 = f.read_decoded_line(wig_file, zipped)
                    int(line2)
            except Exception:
                invalid = True
                warning = f"Tried to look at a provided wig file: {one_wig} and failed to properly read it. Please check the format."
        else:
            invalid = True
            warning = "I don't see any WIG file in conservation reference directory."
    else:
        invalid = True
        warning = 'Provided conservation reference directory does not exist.'

    return warning if invalid else None


def is_ratio(string):
    invalid = False
    if ':' in string and len(string.split(':')) == 4:
        try:
            ints = [int(part) for part in string.split(':')]
        except Exception:
            invalid = True
            warning = 'All parts of the split ratio must be numbers.'
    else:
        invalid = True
        warning = 'Invalid format of the split ratio.'

    return warning if invalid else None


def is_dataset_dir(folder):
    invalid = False

    if os.path.isdir(folder):
        # TODO later check also the metadata file
        files = f.list_files_in_dir(folder)
        dataset_files = {'train': [], 'validation': [], 'test': []}  # TODO later add also blackbox
        for file in files:
            category = next((category for category in dataset_files.keys() if category in file), None)
            if category: dataset_files[category].append(file)
        for category, files in dataset_files.items():
            if len(files) != 1:
                invalid = True
                warning = 'Each category (train, test, validation) must be represented by exactly one preprocessed file in the given folder.'
            else:
                if files[0] != f'{category}.zip':
                    invalid = True
                    warning = ''
    else:
        invalid = True
        warning = 'Given folder with preprocessed files does not exist.'

    return warning if invalid else None


def is_full_dataset(file_path=None, branches=None):
    invalid = False
    try:
        dataset = Dataset.load_from_file(file_path)
        category = dataset.category
        base_columns = ['chrom_name', 'seq_start', 'seq_end', 'strand_sign', 'klass']
        if category or not all(col in dataset.df.columns for col in base_columns):
            invalid = True
            warning = 'Given mapped file does not contain necessary data. Please check the file.'
        if not all(col in dataset.df.columns for col in branches):
            invalid = True
            warning = 'Given mapped file does not contain selected branches.'
    except Exception:
        invalid = True
        warning = 'Sorry, could not parse given mapped file. Please check the file.'

    return warning if invalid else None
