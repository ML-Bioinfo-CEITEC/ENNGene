from optparse import OptionParser, OptionGroup
import sys
import numpy
from re import sub

usage = "usage: %prog [options]"
opt = OptionParser(usage=usage)

encoded_alphabet = None
valid_chrs = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13',
              'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrY', 'chrX', 'chrMT']
dna_complementary = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}


# https://docs.python.org/3/library/optparse.html#generating-help

opt.add_option(
    "--coord", 
    action="store", 
    help="Coordinates BED File, omit for STDIN",
    default="-"
)
opt.add_option(
    "--ref", 
    action="store", 
    help="Path to reference file or folder, omit for STDIN",
    default="-"
)
opt.add_option(
    "--reftype",
    default="fasta",
    help="Reference filetype: fasta or wig or json [default: %default]"
)
opt.add_option(
    "--onehot", 
    action="store", 
    help="If data needs to be converted to One Hot encoding, give list of alphabet used",
    dest="onehot",
    default = "-"
)
opt.add_option(
    "--score", 
    action="store_false", 
    help="If data does not need to be converted",
    dest="onehot"
)
opt.add_option(
    "--strand",
    default=False,
    help="Apply strand information when mapping interval file to reference [default: %default]"
)
opt.add_option(
    "-v", "--verbose",
    action="store_true", 
    dest="verbose", 
    default=True,
    help="make lots of noise [default]"
)
opt.add_option(
    "-q", "--quiet",
    action="store_false", dest="verbose",
    help="be vewwy quiet (I'm hunting wabbits)"
)
(options, args) = opt.parse_args()

if options.onehot == "-":
    opt.error("Either option --onehot or --score needs to be provided")

if options.verbose:
    print "reading %s..." % options.coord


def filehandle_for(filename):
    if filename == "-":
        filehandle = sys.stdin
    else:
        filehandle = open(filename)
    return filehandle


def fasta_to_dictionary(fasta_file):
    file = filehandle_for(fasta_file)
    dict = {}

# TODO generalize for multiline fasta
    while True:
        line1 = file.readline()
        line2 = file.readline()
        if not line2: break  # EOF

        # Make sure each pair starts with valid identifier, so that we do not iterate over nonsense.
        if not '>' in line1:
            raise Exception("Invalid sequence identifier. Please provide a valid Fasta file (with '>' identifier).")

        # Save only sequence for chromosomes we are interested in (skip scaffolds etc.)
        if line1.strip() in valid_chrs:
            dict.update({sub('>', '', line1.strip()): line2.strip()})
    return dict


def complement(sequence, dictionary):
    return ''.join([dictionary[base] for base in sequence])


def bed_to_seq_dictionary(bed_file, ref_dictionary, strand=False):
    #TODO check format of the input file - accept only some particular format, e.g. bed6 ?
    file = filehandle_for(bed_file)
    dict = {}

    #TODO keep any other information from the original bed file? do we need name of the bed line?
    for line in file:
        values = line.split()
        key = values[0] + "_" + values[1] + "_" + values[2] + "_" + values[5]

        if values[0] in ref_dictionary.keys():
            start_position = int(values[1])
            end_position = (int(values[2])-1)
            value = ref_dictionary[values[0]][start_position:end_position]
            if strand and values[5] == '-':
                value = complement(value, dna_complementary)

        if key and value:
            dict.update({key: value.split()})

    return dict


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
        warning = "Invalid character '{}' found. Provided encoding must contain all possible characters (case-insensitive)."
        raise Exception(warning.format(char))


if options.reftype == 'fasta':
    #TODO create the reference dictionary outside of this script - only once for all the files, and pass it as arg
    ref_dict = fasta_to_dictionary(options.ref)
    result_dict = bed_to_seq_dictionary(options.coord, ref_dict, options.strand)


# else:
    # open reference file or folder
    # save in dictionary reference{chromosome}{position} = value (can be sequence or score)
    # open coordinates file
    # coordFH = filehandle_for(options.coord)
    #for line in coordFH:
        # extract relevant positions from reference dictionary
        # values{name} = [values_per_position_in_range]
        # return result_dict

        
#open coordinates file
# make dictionaries: coords{name} = "chr_start_stop_strand" (this is the key)
# 


if options.onehot:
    encoding = encode_alphabet(options.onehot)

    for key, arr in result_dict.items():
        new_arr = [translate(item, encoding) for item in arr]
        result_dict.update({key: new_arr})


# use result_dict wheter it is translated or not and save it to a file or something

#  print output join("\t", key,  join",", [value{name}])


