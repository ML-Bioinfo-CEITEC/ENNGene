# accepts ini file in python

# runs custom combination of separate subprocesses
# just keeps the order in which they are run (probably)

# accepts options for separate subprocess as well as general options (e.g. verbose)

## ONLY USAGE
# deepnet batch_run ini.py


## EXAMPLE

# a = 'sequence'
# b = 'fasta'

# { global_options => {verbose => true},
# preprocess => {classes => ['pos', 'neg'], branches => [a, 'structure'], ref => b},
# train => {main_brach => a},
# predict => {} }