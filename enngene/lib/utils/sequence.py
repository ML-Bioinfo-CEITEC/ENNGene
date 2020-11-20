import re
import streamlit as st
import _io

from . import file_utils as f
from .exceptions import UserInputError

# TODO allow option custom, to be specified by text input
# TODO add amino acid alphabet - in that case disable cons and fold i guess
ALPHABETS = {'DNA': ['A', 'C', 'G', 'T', 'N'],
             'RNA': ['A', 'C', 'G', 'U', 'N']}

COMPLEMENTARY = {'DNA': {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'},
                 'RNA': {'A': 'U', 'C': 'G', 'G': 'C', 'U': 'A', 'N': 'N'}}


@st.cache(hash_funcs={_io.TextIOWrapper: lambda _: None}, suppress_st_warning=True)
def read_and_cache(fasta):
    with st.spinner('Parsing reference fasta file to infer the available chromosomes. Might take up to few minutes...'):
        parsed = parse_fasta_reference(fasta)
    return parsed


def parse_fasta_reference(fasta_file):
    file = f.filehandle_for(fasta_file)
    seq_dict = {}
    chromosomes = []
    alphabet = set()

    key = None
    value = ""

    for line in file:
        if '>' in line:
            # Save finished previous key value pair (unless it's the first iteration)
            if key:  # and is_valid_chr(key):
                # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
                chromosomes.append(key)
                seq_dict.update({key: value.strip()})

            key = line.strip().strip('>')
            value = ""
        else:
            if key:
                line = line.strip()
                value += line
                l = [char for char in line.upper()]
                alphabet.update(l)
            else:
                raise UserInputError("Provided reference file does not start with '>' fasta identifier.")

    # Save the last kay value pair
    if key:  # and is_valid_chr(key):
        chromosomes.append(key)
        seq_dict.update({key: value.strip()})
    file.close()
    chromosomes.sort()

    return seq_dict, chromosomes, alphabet


# def is_valid_chr(chromosome):
#     return not not re.search(r'^(chr)*((\d{1,3})|(M|m|MT|mt|x|X|y|Y))$', chromosome)


def parse_wig_header(line):
    # example: fixedStep chrom=chr22 start=10510001 step=1 # may also contain span (default = 1)
    header = {'span': 1}

    parts = line.split()
    file_type = parts.pop(0)
    header.update({'file_type': file_type})

    if file_type not in ['fixedStep', 'variableStep']:
        raise UserInputError(f'Unknown type of wig file provided: {file_type}. Only fixedStep or variableStep allowed.')

    for part in parts:
        key, value = part.split('=')
        if key == 'chrom':
            header.update({key: value})
        elif key in ['span', 'start', 'step']:
            header.update({key: int(value)})

    return header


def parse_wig_line(line, header):
    parsed_line = {}
    if header['file_type'] == 'variableStep':
        parts = line.split()
        start = parts[0]
        value = float(parts[1])
        for i in range(header['span']):
            coord = start + i
            parsed_line.update({coord: value})
        header['start'] = start + header['span']
    elif header['file_type'] == 'fixedStep':
        value = float(line)
        for i in range(header['span']):
            coord = header['start'] + i
            parsed_line.update({coord: value})
        header['start'] += header['step']

    return [header, parsed_line]


def complement(sequence_list, dictionary):
    return [dictionary[base] for base in sequence_list]


@st.cache
def onehot_encode_alphabet(alphabet):
    class_name = alphabet.__class__.__name__
    if class_name != 'list' and class_name != 'ndarray':
        raise ValueError('Alphabet must be a List. Instead, object of class {class_name} was provided.')

    encoded_alphabet = {}
    for i, char in enumerate(alphabet):
        # array = numpy.zeros([len(alphabet)])
        array = []
        for x in range(len(alphabet)):
            array.append(0.0)
        array[i] = 1.0
        encoded_alphabet.update({str(char).upper(): array})

    return encoded_alphabet


# def dna_to_rna(char):
#     encoding = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U', 'N': 'N'}
#     translated_letter = translate(char, encoding)
#     return translated_letter


def translate(char, encoding):
    if not char:
        return None
    if not encoding or not (encoding.__class__.__name__ == 'dict'):
        raise ValueError('Encoding missing...')

    if char.upper() in encoding.keys():
        return encoding[char.upper()]
    else:
        raise UserInputError(f"Invalid character '{char}' found, given encoding {encoding}. "
                         "Provided encoding must contain all possible characters (case-insensitive).")


def define_alphabet(alphabet):
    if all(x in ALPHABETS['DNA'] for x in alphabet):
        return 'DNA'
    elif all(x in ALPHABETS['RNA'] for x in alphabet):
        return 'RNA'
    else:
        print(alphabet)
        raise UserInputError('The characters used in the reference fasta file do not match DNA nor RNA sequence. Can not continue, sorry.')