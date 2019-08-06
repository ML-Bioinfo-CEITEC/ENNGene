import os
import sys
import logging

from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('main')


class MakeDatasets(Subcommand):

    # TODO allow skipping steps, e.g. datasets separation or size reduction

    def __init__(self, default_args=None):
        help_message = '''deepnet <subcommand> [<args>]
            Preprocess input files by creating datasets containing data specific per each branch. That includes mapping
            to reference, encoding, folding, etc.

            Validation, test and blackbox datasets can be assigned specific chromosomes. The rest of the chromosomes
            is used for training.
            '''

        parser = self.create_parser(help_message)
        super().__init__(parser)

        if default_args is not None:
            self.args = parser.parse_args(default_args)
        else:
            self.args = parser.parse_args(sys.argv[2:])
        logger.info('Running make_datasets with the following arguments: ' + str(self.args)[10:-1])

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

        if self.args.reducelist:
            self.reducelist = self.args.reducelist
            if self.args.reduceratio:
                self.reduceratio = [float(x) for x in self.args.reduceratio]
            if self.args.reduceseed:
                self.reduceseed = self.args.reduceseed
        else:
            self.reducelist = None

        self.split = self.args.split
        if self.split == 'by_chr':
            self.chromosomes = {'validation': self.args.validation,
                                'test': self.args.test,
                                'blackbox': self.args.blackbox,
                                'train': (seq.VALID_CHRS - self.args.validation - self.args.test - self.args.blackbox)}
        elif self.split == 'rand':
            self.split_seed = self.args.splitseed
            self.splitratio_list = self.args.splitratio.split(':')
            if len(self.splitratio_list) != 4:
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
            # TODO allow naming the classes differently?
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
            help="Apply strand information when mapping interval file to reference. [default: False]"
        )
        parser.add_argument(
            "--reducelist",
            nargs='+',
            help="Provide list of classes you wish to reduce (e.g. 'positive' 'negative'). \n \
            Names of the classes must correspond to the input file names."
        )
        parser.add_argument(
            "--reduceratio",
            nargs='+',
            help="Define reducing ratio per each class you wish to reduce (e.g. 0.2 0.5). \n \
            --reducelist must be provided."
        )
        parser.add_argument(
            "--reduceseed",
            help="You may provide seed for random separation of datasets. --split must be set to 'rand'."
        )
        parser.add_argument(
            "--split",
            default='by_chr',
            choices=['by_chr', 'rand'],
            help="Criteria for separation into test, train, validation and blackbox datasets. [default: 'by_chr']"
        )
        parser.add_argument(
            "--splitseed",
            help="You may provide seed for random separation of datasets. --split must be set to 'rand'."
        )
        # TODO what ratio to use as default? What would be better way to define the ratio?
        parser.add_argument(
            "--splitratio",
            default='10:2:2:1',
            help="Ratio for random separation. The order is as follows: train:validation:test:blackbox. \n \
            --split must be set to 'rand'. [default: '10:2:2:1']"
        )
        parser.add_argument(
            "--validation",
            default={'chr19', 'chr20'},
            help="Set of chromosomes to be included in the validation set. --split must be set to 'by_chr'. \
                        [default: {'chr19', 'chr20'}]"
        )
        parser.add_argument(
            "--test",
            default={'chr21'},
            help="Set of chromosomes to be included in the test set. --split must be set to 'by_chr'. \
                 [default: {'chr21'}]"
        )
        parser.add_argument(
            "--blackbox",
            default={'chr22'},
            help="Set of chromosomes to be included in the blackbox set for final evaluation. \
                 --split must be set to 'by_chr'. [default: {'chr22'}]"
        )
        return parser

    def reference(self, branch):
        if branch == 'cons':
            return self.cons_ref
        elif branch == 'seq' or branch == 'fold':
            return self.seq_ref

    def run(self):
        # super().run(self.args)
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
                if branch not in datasets.keys(): datasets.update({branch: set()})
                datasets[branch].add(Dataset(branch, klass=klass, bed_file=file,
                                             ref_dict=self.reference(branch), strand=self.args.strand,
                                             encoding=encoding))

        # TODO export before random selection
        # Randomly select smaller datasets
        if self.reducelist:
            reduced_datasets = {}
            for branch, dsets in datasets.items():
                if branch not in reduced_datasets.keys(): reduced_datasets.update({branch: set()})
                for dataset in dsets:
                    if dataset.klass in self.reducelist:
                        print("reducing klass {} for branch {}".format(dataset.klass, branch))
                        ratio = self.reduceratio[self.reducelist.index(dataset.klass)]
                        reduced_datasets[branch].add(dataset.reduce(ratio, self.reduceseed))
                    else:
                        print("keeping klass {} for branch {}".format(dataset.klass, branch))
                        reduced_datasets[branch].add(dataset)
        else:
            reduced_datasets = datasets

        # Split data into train, validation, test and blackbox datasets
        split_datasets = {}

        for branch, dsets in reduced_datasets.items():
            if branch not in split_datasets.keys(): split_datasets.update({branch: []})
            for dataset in dsets:
                if self.split == 'by_chr':
                    split_subsets = Dataset.split_by_chr(dataset, self.chromosomes)
                elif self.split == 'rand':
                    # TODO export before random separation
                    split_subsets = Dataset.split_random(dataset, self.splitratio_list, self.split_seed)
                split_datasets[branch].append(split_subsets)

        # Separate positives and negatives (classes) within all the categories across the branches
        datasets_to_merge = {}
        for branch, dsets in split_datasets.items():
            if branch not in datasets_to_merge.keys(): datasets_to_merge.update({branch: {}})
            for split_subsets in dsets:
                for category, dataset in split_subsets.items():
                    if category not in datasets_to_merge[branch].keys():
                        datasets_to_merge[branch].update({category: []})
                    datasets_to_merge[branch][category].append(dataset)

        # Merge datasets of the same klass within all the branch (e.g. pos + neg)
        final_datasets = set()
        for branch, dictionary in datasets_to_merge.items():
            for category, list_of_datasets in dictionary.items():
                print(category, list_of_datasets)
                final_datasets.add(Dataset.merge(list_of_datasets))

        # Export final datasets to files
        for dataset in final_datasets:
            branch_folder = os.path.join(self.output_folder, 'datasets', dataset.branch)
            if not os.path.exists(branch_folder): os.makedirs(branch_folder)
            dataset.save_to_file(branch_folder)

        return final_datasets
