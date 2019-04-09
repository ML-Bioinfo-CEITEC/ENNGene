from optparse import OptionParser, OptionGroup
import sys

usage = "usage: %prog [options]"
opt = OptionParser(usage=usage)

# https://docs.python.org/3/library/optparse.html#generating-help

opt.add_option(
    "--coord", 
    action="store", 
    help="Coordinates BED File, ommit for STDIN",
    default="-"
)
opt.add_option(
    "--ref", 
    action="store", 
    help="Path to reference file or folder, ommit for STDIN",
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

def filehandle_for (filename): 
    if filename == "-":
        filehandle = sys.stdin
    else:
        filehandle = open(filename)
    return filehandle

if options.onehot == "-":
    parser.error("Either option --onehot or --score needs to be provided")

if options.verbose:
    print "reading %s..." % options.coord

# if reftype is fasta => use bedtools
    # open produced fasta file
    # extract sequence into dictionary: values{name} = [split("", sequence)]

# else
    # open reference file or folder


    # save in dictionary reference{chromosome}{position} = value (can be sequence or score)

    # open coordinates file

    coordFH = filehandle_for(options.coord)
    for line in coordFH:
        # extract relevant positions from reference dictionary
        # values{name} = [values_per_position_in_range]
        
#open coordinates file
# make dictionaries: coords{name} = "chr_start_stop_strand" (this is the key)
# 

# if options.onehot == FALSE -> we can print output join("\t", key,  join",", [value{name}])
# else:
    # create onehot table
    # foreach item in values: convert values to onehot => print output join("\t", key, join",", [onehot_value{name}])




