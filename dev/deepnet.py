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

        # FIXME replace with check for class / file of that name
        klass = find_class_containing_that_name
        if not klass:
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke object of class with same name as the subcommand
        from .lib import klass
        subcommand = args.command.capitalize()()
        subcommand.run()


if __name__ == '__main__':
    DeepNet()
