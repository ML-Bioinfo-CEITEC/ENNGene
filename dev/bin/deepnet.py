#!/usr/bin/env python

import argparse
import os
import sys

sys.path.append(os.getcwd()+'/lib')


class DeepNet:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='rbp deepnet',
            usage='''deepnet <command> [<args>]
                Some short overall description of the program... Run with:
                    batch_run           Run several subcommands together, or
                    other subcommands   ...
                    
                Usage
                deepnet preprocess --classes ['pos', 'neg'] --branches ['seq1', 'seq2', 'structure'] --test ['chr22']
            ''')

        parser.add_argument('command', help='Subcommand to run')

        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])

        module = __import__("lib.{}".format(args.command))
        if not module:
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # TODO discuss naming and import conventions for the subcommands - probably best would be name of the dir match
        # the subcommand for easier dispatching and to import full dir as a package (use the init file)
        # if possible in python 3.7 ?

        # use dispatch pattern to invoke object of class with same name as the subcommand
        # TODO generalize for multiple-word classes
        subcommand = args.command.capitalize()()
        subcommand.run()


if __name__ == '__main__':
    DeepNet()
