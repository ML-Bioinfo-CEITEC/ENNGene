import sys

from utils import file_utils as f
from utils import sequence as seq
from utils.subcommand import Subcommand

from dataset import Dataset
import separate_sets_by_chr


class Preprocess(Subcommand):

    def __init__(self):
        help_message = '''deepnet <command> [<args>]
            Subcommand-specific description ....    
            '''

        parser = self.initialize_parser(help_message)

        # TODO add options for
        # validation = self.args.validation or ['chr10']
        # test = self.args.test or ['chr20']
        # blackbox = self.args.blackbox or ['chr21']
        # train = seq.VALID_CHRS - validation - test - blackbox

        # TODO accept several input files
        parser.add_argument(
            # arguments without -- are mandatory
            "coord",
            action="store",
            nargs='?',
            help="Coordinates BED File per each class, omit for STDIN",
            default='-'
        )
        parser.add_argument(
            "ref",
            action="store",
            help="Path to reference file or folder, omit for STDIN",
            default="-"
        )
        parser.add_argument(
            "--reftype",
            default="fasta",
            help="Reference filetype: fasta or wig or json [default: %default]"
        )
        parser.add_argument(
            "--branches",
            default="seq",
            help="List of branches delimited by comma. Accepted values are 'seq', 'cons', 'fold'. [default: %default]"
        )
        parser.add_argument(
            # arguments with -- are optional
            "--onehot",
            action="store",
            help="If data needs to be converted to One Hot encoding, give a list of alphabet used.",
            dest="onehot",
            default="-"
        )
        parser.add_argument(
            "--score",
            action="store_false",
            help="If data does not need to be converted",
            dest="onehot"
        )
        parser.add_argument(
            "--strand",
            default=False,
            help="Apply strand information when mapping interval file to reference [default: %default]"
        )

        self.args = parser.parse_args(sys.argv[2:])

        self.encoded_alphabet = None
        self.ref_dict = self.make_ref_dict(self.args.ref, self.args.ref_type)
        self.branches = self.args.branches.split(',')
        self.input_files = self.collect_input_files(self.args)
        self.chromosomes = {'train': self.args.train,
                            'validation': self.args.validation,
                            'test': self.args.test,
                            'blackbox': self.args.blackbox}
        if self.args.verbose:
                print('Running deepnet preprocess with input files {}'.format(self.input_files.join(',')))

    @staticmethod
    def make_ref_dict(ref_file, ref_type):
        if ref_type == 'fasta':
            return seq.fasta_to_dictionary(ref_file)
        else:
            warning = "Unknown reference type. Accepted types are 'fasta' and 'something_else'."
            raise Exception(warning)

    @staticmethod
    def collect_input_files(args):
        return list_of_files

    def run(self):
        super().run(self.args)
        datasets = {}

        # create separate folders

        # one input file per class
        for file in self.input_files:
            # TODO how to dynamically create variable name "seq_{}_dataset".format(klass)

            for branch in self.branches:
                datasets[branch] += [Dataset(file, self.ref_dict, ['A','C','G','T','N']).create(branch, )]
                # > sequence_${class}_dataset.txt

        # Merge positives and negatives (classes) and add labels
        for branch in datasets.keys:
            datasets_to_merge = []
            for dataset in datasets[branch]:
                datasets_to_merge.append(f.table_paste_col(dataset, "class", dataset.klass))
            datasets[branch] = Dataset.merge(datasets_to_merge)

        for branch in datasets.keys:
            for category in self.chromosomes.keys():
                branch_category_dataset = separate_sets_by_chr(datasets[branch], self.chromosomes[category]










