import numpy
import os
from re import sub
from zipfile import ZipFile

from . import file_utils as f


VALID_CHRS = {'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13',
              'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrY', 'chrX', 'chrMT'}
DNA_COMPLEMENTARY = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


def fasta_to_dictionary(fasta_file):
    file = f.filehandle_for(fasta_file)
    seq_dict = {}

    # TODO generalize for multiline fasta
    while True:
        line1 = file.readline()
        line2 = file.readline()
        if not line2: break  # EOF

        # Make sure each pair starts with valid identifier, so that we do not iterate over nonsense.
        if '>' not in line1:
           raise Exception("Invalid sequence identifier. Please provide a valid Fasta file (with '>' identifier).")

        # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
        if line1.strip() in VALID_CHRS:
            seq_dict.update({sub('>', '', line1.strip()): line2.strip()})
    file.close()
    return seq_dict


def wig_to_dictionary(ref_path):
    zipped = f.list_files_in_dir(ref_path, '.wig')
    files = ZipFile.extractall(ref_path, zipped)
    cons_dict = {}

    for file in files:
        file = f.filehandle_for(file)
        for line in file:
            # parse the header line
            # example: fixedStep chrom=chr22 start=10510001 step=1 # may also contain span (default = 1)
            if 'chrom' in line:
                chr = None
                start = None
                step = None
                step_no = 0
                span = 1

                parts = line.split()
                file_type = parts.pop(0)
                if file_type not in ['fixedStep', 'variableStep']:
                    warning = "Unknown type of wig file provided: {}. Only fixedStep or variableStep allowed."
                    raise Exception(warning.format(file_type))

                for part in parts:
                    key, value = part.split('=')
                    if key == 'chrom':
                        chr = value
                    elif key == 'span':
                        span = int(value)
                    elif key == 'start':  # only for fixedStep
                        start = int(value)
                    elif key == 'step':  # only for fixedStep
                        step = int(value)

                if chr not in cons_dict.keys(): cons_dict.update({chr: {}})

            # update values, until another header is met
            else:
                if not chr: next
                if file_type == 'variableStep':
                    parts = line.split()
                    start = parts[0]
                    value = int(parts[1])
                    for i in range(span):
                        coord = start + i
                        cons_dict[chr].update({coord: value})
                elif file_type == 'fixedStep':
                    value = int(line)
                    for i in range(span):
                        coord = start + (step_no * step) + i
                        cons_dict[chr].update({coord: value})
                    step_no += 1

        file.close()
    # dictionary format: {chr => {coordinate => int_value}}
    return cons_dict


def complement(sequence, dictionary):
    return ''.join([dictionary[base] for base in sequence])


def encode_alphabet(alphabet, force_new=False):
    global encoded_alphabet

    if encoded_alphabet and not force_new:
        return encoded_alphabet
    else:
        class_name = alphabet.__class__.__name__
        if class_name != 'list':
            raise Exception('Alphabet must be a List. Instead, object of class {} was provided.'.format(class_name))

        encoded_alphabet = {}
        for i, char in enumerate(alphabet):
            array = numpy.zeros([len(alphabet)])
            array[i] = 1.0
            encoded_alphabet.update({str(char).lower(): array})

    return encoded_alphabet


def translate(char, encoding):
    if not char:
        return None
    if not encoding or not (encoding.__class__.__name__ == 'dict'):
        raise Exception('')

    if char.lower() in encoding.keys():
        return encoding[char.lower()]
    else:
        warning = "Invalid character '{}' found. " \
                  "Provided encoding must contain all possible characters (case-insensitive)."
        raise Exception(warning.format(char))