#!/usr/bin/env python

import argparse
import os
import sys
import logging

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.append(os.getcwd())
logging.basicConfig(filename='app.log',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

logger = logging.getLogger('main')
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',  datefmt='%m/%d/%Y %I:%M:%S %p')
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)


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
        logger.info('DeepNet started with the following subcommands: ' + str(sys.argv[1:2]))

        module_path = "lib.{}.{}".format(args.subcommand, args.subcommand)
        subcommand_class = dirname_to_class(args.subcommand)
        try:
            module = __import__(module_path, fromlist=[subcommand_class])
            logger.info('Module ' + args.subcommand + ' successfully imported.')
        except ModuleNotFoundError:
            logger.error('Unrecognized subcommand.')
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
