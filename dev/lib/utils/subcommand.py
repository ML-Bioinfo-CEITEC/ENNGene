import argparse


class Subcommand:

    def run(self, args):
        # TODO define generic parts of runnign the subcommand
        return args

    # private method for Subcommand descendants use only
    @staticmethod
    def initialize_parser(subcommand_help):
        parser = argparse.ArgumentParser(
            description='rbp deepnet',
            usage=subcommand_help
        )
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

