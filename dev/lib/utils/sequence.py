from re import sub
import numpy

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
    return seq_dict


def complement(sequence, dictionary):
    return ''.join([dictionary[base] for base in sequence])


def bed_to_seq_dictionary(bed_file, ref_dictionary, strand=False):
    # TODO check format of the input file - accept only some particular format, e.g. bed6 ?
    file = f.filehandle_for(bed_file)
    seq_dict = {}

    # TODO keep any other information from the original bed file? do we need name of the bed line?
    for line in file:
        values = line.split()

        # 0 - chr. name, 1 - seq. start, 2 - seq. end, 5 - strand
        key = values[0] + "_" + values[1] + "_" + values[2] + "_" + values[5]

        if values[0] in ref_dictionary.keys():
            start_position = int(values[1])
            end_position = (int(values[2])-1)
            sequence = ref_dictionary[values[0]][start_position:end_position]
            if strand and values[5] == '-':
                sequence = complement(sequence, DNA_COMPLEMENTARY)

        if key and sequence:
            seq_dict.update({key: sequence.split()})

    return seq_dict


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