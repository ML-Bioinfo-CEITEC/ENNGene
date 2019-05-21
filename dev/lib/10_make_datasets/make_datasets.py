import os
import sys

from ..utils import file_utils as f
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

from .dataset import Dataset


class MakeDatasets(Subcommand):

    def __init__(self):
        help_message = '''deepnet <command> [<args>]
            Subcommand-specific description ....    
            
            Validation, test and blackbox datasets can be assigned specific chromosomes. The rest of the chromosomes 
            shall be used for training.
            '''

        parser = self.initialize_parser(help_message)

        # TODO accept several input files
        # TODO somehow define acceptable values per attribute
        # TODO check format of the input file - accept only some particular format, e.g. bed6 ?
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
            default=["seq"],
            help="List of branches delimited by comma. Accepted values are 'seq', 'cons', 'fold'. [default: %default]"
        )
        parser.add_argument(
            # arguments with -- are optional
            "--onehot",
            action="store",
            help="If data needs to be converted to One Hot encoding, give a list of alphabet used.",
            dest="onehot",
            default=None
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
        parser.add_argument(
            "--validation",
            default={'chr19', 'chr20'},
            help="Set of chromosomes to be included in the validation set [default: %default]"
        )
        parser.add_argument(
            "--test",
            default={'chr21'},
            help="Set of chromosomes to be included in the test set [default: %default]"
        )
        parser.add_argument(
            "--blackbox",
            default={'chr22'},
            help="Set of chromosomes to be included in the blackbox set for final evaluation [default: %default]"
        )

        self.args = parser.parse_args(sys.argv[2:])

        self.encoded_alphabet = None
        self.ref_dict = self.make_ref_dict(self.args.ref, self.args.ref_type)
        self.branches = self.args.branches.split(',')
        self.input_files = self.collect_input_files(self.args)
        self.chromosomes = {'validation': self.args.validation,
                            'test': self.args.test,
                            'blackbox': self.args.blackbox,
                            'train': (seq.VALID_CHRS - self.args.validation - self.args.test - self.args.blackbox)}
        if self.args.verbose:
            print('Running deepnet preprocess with input files {}'.format(self.input_files.join(',')))

    @staticmethod
    def make_ref_dict(ref_path, ref_type):
        if ref_type == 'fasta':
            return seq.fasta_to_dictionary(ref_path)
        elif ref_type == 'wig':
            return seq.wig_to_dictionary(ref_path)
        else:
            warning = "Unknown reference type. Accepted types are 'fasta', 'wig' and 'something_else?'."
            raise Exception(warning)

    @staticmethod
    def collect_input_files(args):
        return list_of_files

    def run(self):
        super().run(self.args)
        datasets = {}

        if self.args.onehot:
            encoding = seq.encode_alphabet(self.args.onehot)
        else:
            encoding = None

        # Accept one file per class nad then generate requested branches from that
        for file in self.input_files:
            for branch in self.branches:
                if not datasets[branch]: datasets[branch] = []
                datasets[branch] += [Dataset(branch).create(file, self.ref_dict, self.args.strand, encoding)]

        # Merge positives and negatives (classes) and add labels
        for branch in datasets.keys:
            datasets_to_merge = []
            for dataset in datasets[branch]:
                datasets_to_merge.append(f.table_paste_col(dataset, "class", dataset.klass))
            datasets[branch] = Dataset.merge(datasets_to_merge)

        for branch in datasets.keys:
            for category in self.chromosomes.keys():
                branch_category_dataset = separate_sets_by_chr(datasets[branch], self.chromosomes[category]










