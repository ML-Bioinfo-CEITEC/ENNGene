import argparse
from datetime import datetime
import os
import sys


class Subcommand:

    def __init__(self, parser):
        self.args = parser.parse_args(sys.argv[2:])

        if self.args.output:
            self.output_folder = self.args.output
            if not os.path.exists(self.output_folder):
                os.makedirs(self.args.output)
        else:
            self.output_folder = os.path.join(os.getcwd(), 'output')
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

        self.verbose = self.args.verbose

    # def run(self, args):
        # TODO define generic parts of running the subcommand
        # pass

    # private method for Subcommand descendants use only
    @staticmethod
    def initialize_parser(subcommand_help):
        parser = argparse.ArgumentParser(
            description='rbp deepnet',
            usage=subcommand_help
        )
        parser.add_argument(
            "-o", "--output",
            dest='output',
            help="Specify folder for output files. If not specified, current working directory will be used."
        )
        # TODO do we need verbose and quite, or rather use the logger for everything?
        # Or use it to set level of logger verbosity?
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            dest="verbose",
            default=True,
            help="make lots of noise [default]"
        )
        parser.add_argument(
            "-q", "--quiet",
            action="store_false",
            dest="verbose",
            help="be vewwy quiet (I'm hunting wabbits)"
        )
        return parser

    @staticmethod
    def spent_time(time1):
        time2 = datetime.now()
        t = time2 - time1
        return t
