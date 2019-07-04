import os
import sys
import logging

from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('main')


class MakeDatasets(Subcommand):

    def __init__(self, default_args):
        help_message = '''deepnet <subcommand> [<args>]
            Preprocess input files by creating datasets containing data specific per each branch. That includes mapping
            to reference, encoding, folding, etc.

            Validation, test and blackbox datasets can be assigned specific chromosomes. The rest of the chromosomes
            is used for training.
            '''

        parser = self.create_parser(help_message)
        if default_args is not None:
            self.args = parser.parse_args(default_args)
        else:
            self.args = parser.parse_args(sys.argv[2:])
        logger.info('Running make_datasets with the following arguments: ' + str(self.args)[10:-1])

        if self.args.output:
            self.output_folder = self.args.output
            if not os.path.exists(self.output_folder):
                os.makedirs(self.args.output)
        else:
            self.output_folder = os.path.join(os.getcwd(), 'output')
            os.makedirs(self.output_folder)

        self.encoded_alphabet = None
        self.seq_ref = seq.fasta_to_dictionary(self.args.ref)
        self.branches = self.args.branches
        if 'cons' in self.branches:
            if not self.args.consdir:
                logger.exception('Exception occurred.')
                raise Exception("Provide conservation directory for calculating scores for conservation branch.")
            else:
                self.cons_ref = seq.wig_to_dictionary(self.args.consdir)

        if self.args.coord:
            self.input_files = self.args.coord
        else:
            logger.exception('Exception occurred.')
            raise Exception("Input coordinate (.bed) files are required. Provide one file per class.")

        self.separate = self.args.separate
        if self.separate == 'by_chr':
            self.chromosomes = {'validation': self.args.validation,
                                'test': self.args.test,
                                'blackbox': self.args.blackbox,
                                'train': (seq.VALID_CHRS - self.args.validation - self.args.test - self.args.blackbox)}
        elif self.separate == 'rand':
            self.seed = self.args.seed
            self.ratio_list = self.args.ratio.split(':')
            if len(self.ratio_list) != 4:
                logger.exception('Exception occurred.')
                raise Exception("Provide ratio for all four categories (train, validation, test and blackbox). \
                                 Default: '10:2:2:1'.")

        logger.info('Running deepnet make_datasets with input files {}'.format(', '.join(self.input_files)))

    def create_parser(self, message):
        parser = self.initialize_parser(message)

        # TODO allow multiple files per class?
        parser.add_argument(
            "--coord",
            action="store",
            required=True,
            nargs='+',
            help="Coordinates BED File per each class, omit for STDIN, class name = file name",
            default='-'
        )
        parser.add_argument(
            "--ref",
            action="store",
            required=True,
            help="Path to reference file or folder, omit for STDIN",
            default="-"
        )
        parser.add_argument(
            "--branches",
            choices=['seq', 'cons', 'fold'],
            nargs='+',
            default='seq',
            help="Branches. [default: 'seq']"
        )
        parser.add_argument(
            "--consdir",
            action="store",
            help="Directory containing wig files with scores. Necessary if 'cons' branch is selected."
        )
        parser.add_argument(
            "--onehot",
            action="store",
            nargs='+',
            help="If data needs to be converted to One Hot encoding, provide a list of alphabet used.",
            dest="onehot",
            default=None
        )
        parser.add_argument(
            "--strand",
            default=False,
            help="Apply strand information when mapping interval file to reference [default: False]"
        )
        parser.add_argument(
            "--separate",
            default='by_chr',
            choices=['by_chr', 'rand'],
            help="Criteria for separation into test, train, validation and blackbox datasets. [default: 'by_chr']"
        )
        parser.add_argument(
            "--seed",
            help="You may provide seed for random separation of datasets. --separate must be set to 'rand'."
        )
        # TODO what ratio to use as default? What would be better way to define the ratio?
        parser.add_argument(
            "--ratio",
            default='10:2:2:1',
            help="Ratio for random separation. The order is as follows: train:validation:test:blackbox. \n \
            --separate must be set to 'rand'. [default: '10:2:2:1']"
        )
        parser.add_argument(
            "--validation",
            default={'chr19', 'chr20'},
            help="Set of chromosomes to be included in the validation set. --separate must be set to 'by_chr'. \
                        [default: {'chr19', 'chr20'}]"
        )
        parser.add_argument(
            "--test",
            default={'chr21'},
            help="Set of chromosomes to be included in the test set. --separate must be set to 'by_chr'. \
                 [default: {'chr21'}]"
        )
        parser.add_argument(
            "--blackbox",
            default={'chr22'},
            help="Set of chromosomes to be included in the blackbox set for final evaluation. \
                 --separate must be set to 'by_chr'. [default: {'chr22'}]"
        )
        return parser

    def reference(self, branch):
        if branch == 'cons':
            return self.cons_ref
        elif branch == 'seq' or branch == 'fold':
            return self.seq_ref

    def run(self):
        super().run(self.args)
        datasets = {}

        if self.args.onehot:
            encoding = seq.onehot_encode_alphabet(self.args.onehot)
        else:
            encoding = None

        # Accept one file per class nad then generate requested branches from that
        for file in self.input_files:
            file_name = os.path.basename(file)
            if '.bed' in file_name:
                klass = file_name.replace('.bed', '')
            else:
                klass = file_name

            for branch in self.branches:
                if branch not in datasets.keys(): datasets.update({branch: {}})
                datasets[branch].update({klass: Dataset(branch, klass=klass, bed_file=file,
                                                        ref_dict=self.reference(branch), strand=self.args.strand,
                                                        encoding=encoding)})

        # Merge positives and negatives (classes)
        for branch in datasets.keys():
            datasets.update({branch: Dataset.merge(list(datasets[branch].values()))})

        # Separate data into train, validation, test and blackbox datasets
        separated_datasets = {}
        valid_data = []
        for branch in datasets.keys():
            if self.separate == 'by_chr':
                separated_subsets = Dataset.separate_by_chr(datasets[branch], self.chromosomes)
            elif self.separate == 'rand':
                separated_subsets = Dataset.separate_random(datasets[branch], self.ratio_list, self.seed)
            separated_datasets.update({branch: separated_subsets})
            valid_data.append(self.chromosomes)

        for branch, dictionary in separated_datasets.items():
            branch_folder = os.path.join(self.output_folder, 'datasets', branch)
            if not os.path.exists(branch_folder): os.makedirs(branch_folder)
            for category, dataset in dictionary.items():
                dataset.save_to_file(branch_folder, category)

        # Final datasets dictionary in format {branch: {'train': dataset, 'test': dataset2, ...}}
        return separated_datasets, valid_data
