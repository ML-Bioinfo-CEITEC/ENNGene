import numpy as np
import streamlit as st
import _io

from . import file_utils as f
from .exceptions import UserInputError

# TODO allow option custom, to be specified by text input
# TODO add amino acid alphabet - in that case disable cons and fold i guess
ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
FOLDING = {'.': 0, '|': 1, 'x': 2, '<': 3, '>': 4, '(': 5, ')': 6}


@st.cache(hash_funcs={_io.TextIOWrapper: lambda _: None}, suppress_st_warning=True)
def read_and_cache(fasta):
    with st.spinner('Parsing reference fasta file to infer the available chromosomes. May take up to few minutes...'):
        parsed = parse_fasta_reference(fasta)
    return parsed


def parse_fasta_reference(fasta_file):
    chromosomes = []

    key = None
    value = ""

    file, zipped = f.unzip_if_zipped(fasta_file)
    while True:
        line = f.read_decoded_line(file, zipped)
        if not line:
            break
    
        if '>' in line:
            # Save finished previous key value pair (unless it's the first iteration)
            if key:  # and is_valid_chr(key):
                # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
                chromosomes.append(key)

            key = line.strip().strip('>')
            value = ""
        else:
            if key:
                line = line.strip()
                value += line
                l = [char for char in line.upper()]
            else:
                raise UserInputError("Provided reference file does not start with '>' fasta identifier.")

    chromosomes.append(key)  # save the last one
    file.close()
    chromosomes.sort()

    return chromosomes


# def is_valid_chr(chromosome):
#     return not not re.search(r'^(chr)*((\d{1,3})|(M|m|MT|mt|x|X|y|Y))$', chromosome)


def chrom_sizes(sizes_file):
    chrom_sizes = {}
    with open(sizes_file, 'r') as sizes:
        for i, line in enumerate(sizes.readlines()):
            chrom, size = line.strip().split('\t')
            chrom_sizes[chrom] = int(size)

    return chrom_sizes


def wigfile_to_scores(cons_file, chrom_size, out_file):
    # Expects one file containing one chromosome only

    with open(cons_file, 'r') as inf:
        for i, line in enumerate(inf):
            if i == 0:
                file_type, chrom, start, span, step = parse_wig_header(line)
                position = start

                scores_array = np.zeros(chrom_size)
            else:
                if 'chrom' in line:
                    file_type, chrom, start, span, step = parse_wig_header(line)
                    position = start
                else:
                    scores_array, position = parse_wig_line(line, file_type, step, span, position, scores_array)

        np.save(out_file, scores_array)

    return out_file


def parse_wig_header(line):
    # example: fixedStep chrom=chr22 start=10510001 step=1 # may also contain span (default = 1)
    span = 1; start = None; step = None

    parts = line.split()
    file_type = parts.pop(0)

    if file_type not in ['fixedStep', 'variableStep']:
        raise UserInputError(f'Unknown type of wig file provided: {file_type}. Only fixedStep or variableStep allowed.')

    for part in parts:
        key, value = part.split('=')
        if key == 'chrom':
            chrom = value
        elif key == 'start':
            start = int(value) - 1
        elif key == 'span':
            span = int(value)
        elif key == 'step':
            step = int(value)

    return [file_type, chrom, start, span, step]


def parse_wig_line(line, file_type, step, span, position, scores_array):
    if file_type == 'variableStep':
        parts = line.split()
        start = int(parts[0]) - 1
        value = float(parts[1].strip())
        for i in range(span):
            coord = start + i
            scores_array[coord] = value

    elif file_type == 'fixedStep':
        value = float(line.strip())
        for i in range(span):
            coord = position + i
            scores_array[coord] = value
        position += step

    return scores_array, position


def complement(sequence_list, dictionary):
    return [dictionary[base] for base in sequence_list]


def onehot_encode_alphabet(alphabet):
    encoded_alphabet = {}
    for char, pos in alphabet.items():
        ohe = np.zeros(len(set(alphabet.values())))
        ohe[pos] = 1
        encoded_alphabet.update({char: ohe})
    if alphabet == ALPHABET:
        encoded_alphabet.update({'N': np.full(len(set(alphabet.values())), 0.25)})

    return encoded_alphabet


def translate(char, encoding):
    if not char:
        return None
    if not encoding or not (encoding.__class__.__name__ == 'dict'):
        raise ValueError('Encoding missing...')

    if char in encoding.keys():
        return encoding[char]
    elif char.upper() in encoding.keys():
        return encoding[char.upper()]
    else:
        raise UserInputError(f"Invalid character '{char}' found, given encoding {encoding}. "
                         "Provided encoding must contain all possible characters (case-insensitive).")
