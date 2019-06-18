import numpy
import os
from re import sub
import subprocess


from . import file_utils as f

VALID_CHRS = {'chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
              'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrY', 'chrX',
              'chrMT'}
DNA_COMPLEMENTARY = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


# FIXME incorporate usage of the index file for some reasonable run time?
def fasta_to_dictionary(fasta_file):
    file = f.filehandle_for(fasta_file)
    seq_dict = {}

    key = None
    value = ""

    for line in file:
        if '>' in line:
            # Save finished previous key value pair (unless it's the first iteration)
            if key and key.strip() in VALID_CHRS:
                # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
                seq_dict.update({sub('>', '', key.strip()): value.strip()})

            key = line
            value = ""
        else:
            if key:
                value += line.strip()
            else:
                raise Exception("Please provide a valid Fasta file (with '>' identifier).")

    # Save the last kay value pair
    seq_dict.update({sub('>', '', key.strip()): value.strip()})
    
    file.close()
    return seq_dict


def dictionary_to_fasta(dictionary, path, name):
    filepath = os.path.join(path, (name + ".fa"))
    content = ""
    for key, seq in dictionary.items():
        line1 = ">" + key + "\n"
        line2 = seq + "\n"
        content += line1
        content += line2

    f.write(filepath, content.strip())
    # TODO add some check the file was created ok
    return filepath


def wig_to_dictionary(ref_path):
    # TODO for now converting everything without taking VALID_CHRS into account
    zipped = f.list_files_in_dir(ref_path, '.wig')
    files = []
    for zipped_file in zipped:
        files.append(f.unzip_if_zipped(zipped_file))

    cons_dict = {}
    for file in files:
        chr = None
        for line in file:
            # parse the header line
            # example: fixedStep chrom=chr22 start=10510001 step=1 # may also contain span (default = 1)
            if line.__class__.__name__ == 'str' and 'chrom' in line:
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
                if not chr: break
                if file_type == 'variableStep':
                    parts = line.split()
                    start = parts[0]
                    value = float(parts[1])
                    for i in range(span):
                        coord = start + i
                        cons_dict[chr].update({coord: value})
                elif file_type == 'fixedStep':
                    value = float(line)
                    for i in range(span):
                        coord = start + (step_no * step) + i
                        cons_dict[chr].update({coord: value})
                    step_no += 1
        file.close()

    # dictionary format: {chr => {coordinate => int_value}}
    return cons_dict


def complement(sequence_list, dictionary):
    return [dictionary[base] for base in sequence_list]


def onehot_encode_alphabet(alphabet):
    class_name = alphabet.__class__.__name__
    if class_name != 'list':
        raise Exception('Alphabet must be a List. Instead, object of class {} was provided.'.format(class_name))

    encoded_alphabet = {}
    for i, char in enumerate(alphabet):
        array = numpy.zeros([len(alphabet)])
        array[i] = 1.0
        encoded_alphabet.update({str(char).lower(): array})

    return encoded_alphabet


def dna_to_rna(char):
    # TODO is that really all that's necessary? shouldn't it be maid complementary or something?
    encoding = {'A': 'A', 'C': 'C', 'G': 'G', 'T': 'U', 'N': 'N'}
    translated_letter = translate(char, encoding)
    return translated_letter


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


# TODO place the method somewhere else? divide it more?
def fold(dict, name, dna=True):
    if dna:
        rna_dict = {}
        for key, seq in dict.items():
            new_seq = [dna_to_rna(char) for char in seq]
            rna_dict.update({key: new_seq})
    else:
        rna_dict = dict

    # TODO decide where to save the intermediate files
    path = os.getcwd()
    fasta_file = f.dictionary_to_fasta(rna_dict, path, name)

    # TODO without --noconv it substitutes T > U, maybe the dna to rna conversion beforehand is than unnecessary?
    folded_file = os.path.join(path, name + "folded")
    subprocess.run("RNAfold -i {} --jobs=10 --noPS --noconv -o {}".format(fasta_file, folded_file), check=True, )

    return folded_file
