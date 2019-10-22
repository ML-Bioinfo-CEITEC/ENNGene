import argparse
import sys


class BatchRun:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Batch run of several subcommands',
            usage='''deepnet batchrun <command> [<args>]
                Runs custom combination of separate subprocesses.
                Accepts ini file in python with options for separate subprocess 
                as well as extra options describing the batch run.
                    
                Usage
                deepnet batch_run <command> ini.py
            ''')

        parser.add_argument('command', help='Batch subcommand to run')
        args = parser.parse_args(sys.argv[2:3])

        module = __import__(f'lib.batch_run.{args.command}')
        if not module:
            print('Unrecognized batch command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke object of class with same name as the subcommand
        # TODO generalize for multiple-word classes
        subcommand = args.command.capitalize()()
        subcommand.run()


## Ini.py example

# a = 'sequence'
# b = 'fasta'

# { global_options => {verbose => true},
# preprocess => {classes => ['pos', 'neg'], branches => [a, 'structure'], ref => b},
# train => {main_brach => a},
# predict => {} }