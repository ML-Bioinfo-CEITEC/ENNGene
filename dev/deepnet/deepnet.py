#!/usr/bin/env python

import argparse
import os
import sys

sys.path.append(os.getcwd())


class DeepNet:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='rbp deepnet',
            usage='''deepnet <subcommand> [<args>]
                Some short overall description of the program... Run with:
                    batch_run           Run several subcommands together, or
                    other subcommands   ...
                    
                Usage
                deepnet preprocess --classes ['pos', 'neg'] --branches ['seq1', 'seq2', 'structure'] --test ['chr22']
            ''')

        parser.add_argument('subcommand', help='Subcommand to run')

        # parse_args defaults to [1:] for args, but exclude the rest of the args, or validation will fail
        args = parser.parse_args(sys.argv[1:2])

        module_path = "lib.{}.{}".format(args.subcommand, args.subcommand)
        subcommand_class = dirname_to_class(args.subcommand)
        try:
            module = __import__(module_path, fromlist=[subcommand_class])
        except ModuleNotFoundError:
            print('Unrecognized subcommand')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke object of class with same name as the subcommand
        subcommand = getattr(module, subcommand_class)
        subcommand().run()


def dirname_to_class(dirname):
    parts = dirname.split('_')
    return ''.join(x.title() for x in parts)


if __name__ == '__main__':
    DeepNet()
