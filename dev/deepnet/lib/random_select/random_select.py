import logging
import sys

from ..utils.subcommand import Subcommand

logger = logging.getLogger('main')


class RandomSelect(Subcommand):

    def __init__(self):
        help_message = '''deepnet <subcommand> [<args>]
            Randomly pick a subset of the data to reduce training time.
            '''

        parser = self.create_parser(help_message)
        self.args = parser.parse_args(sys.argv[2:])

    def create_parser(self, message):
        parser = self.initialize_parser(message)

        # TODO allow multiple files per class?
        parser.add_argument(
            "--datadir",
            action="store",
            required=True,
            help="Dataset files created by MakeDatasets, omit for STDIN.",
            default='-'
        )

        return parser
