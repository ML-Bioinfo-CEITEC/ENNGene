from copy import deepcopy
import logging
from re import sub

from . import file_utils as f

logger = logging.getLogger('main')

VALID_CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
              'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrY', 'chrX',
              'chrM', 'chrMT']
DNA_COMPLEMENTARY = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def fasta_to_dictionary(fasta_file):
    file = f.filehandle_for(fasta_file)
    seq_dict = {}

    key = None
    value = ""

    for line in file:
        if '>' in line:
            # Save finished previous key value pair (unless it's the first iteration)
            if key in VALID_CHRS:
                # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
                seq_dict.update({key: value.strip()})

            key = line.strip().strip('>')
            value = ""
        else:
            if key:
                value += line.strip()
            else:
                logger.exception('Exception occurred.')
                raise Exception("Please provide a valid Fasta file (with '>' identifier).")

    # Save the last kay value pair
    seq_dict.update({sub('>', '', key.strip()): value.strip()})

    file.close()
    return seq_dict


def parse_wig_header(line):
    # example: fixedStep chrom=chr22 start=10510001 step=1 # may also contain span (default = 1)
    header = {'span': 1}

    parts = line.split()
    file_type = parts.pop(0)
    header.update({'file_type': file_type})

    if file_type not in ['fixedStep', 'variableStep']:
        logger.exception('Exception occurred.')
        raise Exception(
            f'Unknown type of wig file provided: {file_type}. Only fixedStep or variableStep allowed.')

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


def onehot_encode_alphabet(alphabet):
    class_name = alphabet.__class__.__name__
    if class_name != 'list' and class_name != 'ndarray':
        logger.exception('Exception occurred.')
        raise Exception(f'Alphabet must be a List. Instead, object of class {class_name} was provided.')

    encoded_alphabet = {}
    for i, char in enumerate(alphabet):
        # array = numpy.zeros([len(alphabet)])
        array = []
        for x in range(len(alphabet)):
            array.append(0.0)
        array[i] = 1.0
        encoded_alphabet.update({str(char).lower(): array})

    return encoded_alphabet


def dna_to_rna(char):
    encoding = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U', 'N': 'N'}
    translated_letter = translate(char, encoding)
    return translated_letter


def translate(char, encoding):
    if not char:
        return None
    if not encoding or not (encoding.__class__.__name__ == 'dict'):
        logger.exception('Exception occurred.')
        raise Exception('')

    if char.lower() in encoding.keys():
        return encoding[char.lower()]
    else:
        warning = f"Invalid character '{char}' found. " \
                  "Provided encoding must contain all possible characters (case-insensitive)."
        logger.exception('Exception occurred.')
        raise Exception(warning)
