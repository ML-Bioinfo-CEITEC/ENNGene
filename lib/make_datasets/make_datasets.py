from datetime import datetime
import os
import sys
import logging

from ..utils.dataset import Dataset
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('main')


class MakeDatasets(Subcommand):

    # TODO allow skipping steps during batch run, instead use saved files

    def __init__(self, default_args=None):
        help_message = '''deepnet <subcommand> [<args>]
            Preprocess input files by creating datasets containing data specific per each branch. That includes mapping
            to reference, encoding, folding, etc.

            Validation, test and blackbox datasets can be assigned specific chromosomes. The rest of the chromosomes
            is used for training.
            '''

        self.total_time1 = datetime.now()
        parser = self.create_parser(help_message)
        super().__init__(parser)

        if default_args is not None:
            self.args = parser.parse_args(default_args)
        else:
            self.args = parser.parse_args(sys.argv[2:])
        logger.info('Running make_datasets with the following arguments: ' + str(self.args)[10:-1])

        self.encoded_alphabet = None
        self.window = self.args.win
        self.winseed = self.args.winseed
        self.strand = self.args.strand

        self.branches = self.args.branches
        self.references = {}
        if 'seq' in self.branches or 'fold' in self.branches:
            if not self.args.ref:
                logger.exception('Exception occurred.')
                raise Exception("Provide reference fasta file to map interval files to.")
            else:
                seq_dictionary = seq.fasta_to_dictionary(self.args.ref)
                self.references.update({'seq': seq_dictionary, 'fold': seq_dictionary})

        if 'cons' in self.branches:
            if not self.args.consdir:
                logger.exception('Exception occurred.')
                raise Exception("Provide conservation directory for calculating scores for conservation branch.")
            else:
                self.references.update({'cons': self.args.consdir})

        if self.args.coord:
            self.input_files = self.args.coord
        else:
            logger.exception('Exception occurred.')
            raise Exception("Input coordinate (.bed) files are required. Provide one file per class.")

        if self.args.reducelist:
            self.reducelist = self.args.reducelist
            if self.args.reduceratio:
                self.reduceratio = [float(x) for x in self.args.reduceratio]
            else:
                logger.exception('Exception occurred.')
                raise Exception("To reduce selected klasses you must provide reduce ratio per each such klass.")
            self.reduceseed = self.args.reduceseed
        else:
            self.reducelist = None

        self.split = self.args.split
        if self.split == 'by_chr':
            self.chromosomes = {'validation': self.args.validation,
                                'test': self.args.test,
                                'blackbox': self.args.blackbox,
                                'train': (set(seq.VALID_CHRS) - self.args.validation - self.args.test - self.args.blackbox)}
        elif self.split == 'rand':
            self.split_seed = self.args.splitseed
            self.splitratio_list = self.args.splitratio.split(':')
            if len(self.splitratio_list) != 4:
                logger.exception('Exception occurred.')
                raise Exception("Provide ratio for all four categories (train, validation, test and blackbox). \
                                 Default: '10:2:2:1'.")

        logger.debug('Finished initialization.')
        logger.info(f"Running deepnet make_datasets with input files {', '.join(self.input_files)}")

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
            "--win",
            required=True,
            help="Window size to unify lenghts of the input sequences. Default = 100.",
            default=100,
            type=int
        )
        parser.add_argument(
            "--winseed",
            default=64,
            help="Seed to replicate window placement upon the sequences. Default = 64.",
        )
        parser.add_argument(
            "--ref",
            action="store",
            help="Path to reference fasta file. Necessary if 'seq' or 'fold' branch is selected.",
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
            default=112,
            help="You may provide seed for random separation of datasets. --split must be set to 'rand'. Default = 112."
        )
        parser.add_argument(
            "--split",
            default='by_chr',
            choices=['by_chr', 'rand'],
            help="Criteria for separation into test, train, validation and blackbox datasets. [default: 'by_chr']"
        )
        parser.add_argument(
            "--splitseed",
            default=56,
            help="You may provide seed for random separation of datasets. --split must be set to 'rand'. Default = 56."
        )
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

    def run(self):
        if self.args.onehot:
            logger.debug('Encoding alphabet...')
            encoding = seq.onehot_encode_alphabet(self.args.onehot)
        else:
            encoding = None

        # Accept one file per class and create one Dataset per each
        initial_datasets = set()
        logger.debug('Reading in given interval files and applying window...')
        for file in self.input_files:
            file_name = os.path.basename(file)
            allowed_extensions = ['.bed', '.narrowPeak']
            # TODO simpler way to write this in Python?
            # TODO add other possible similar extensions, .bed12 or something like that?
            if any(ext in file_name for ext in allowed_extensions):
                for ext in allowed_extensions:
                    if ext in file_name:
                        klass = file_name.replace(ext, '')
            else:
                logger.exception('Exception occurred.')
                raise Exception(f"Only files of following format are allowed: {', '.join(allowed_extensions)}.")

            initial_datasets.add(
                Dataset(klass=klass, branches=self.branches, bed_file=file, win=self.window, winseed=self.winseed))

        # Merging datapoints from all klasses to map them more quickly all together at once
        all_datapoints = []
        for dataset in initial_datasets:
            all_datapoints += dataset.datapoint_list

        logger.debug(
            f'Mapping intervals from all classes to {len(self.branches)} branch(es) and exporting...')
        merged_dataset = Dataset(branches=self.branches, datapoint_list=all_datapoints)
        dir_path = os.path.join(self.output_folder, 'datasets', 'full_datasets')
        self.ensure_dir(dir_path)
        outfile_path = os.path.join(dir_path, 'merged_all')

        # First ensure order of the DataPoints by chr_name and seq_start within, mainly for conservation
        merged_dataset.sort_datapoints().map_to_branches(
            self.references, encoding, self.strand, outfile_path, self.ncpu)

        mapped_datasets = set()
        sorted_datapoints = {}
        for datapoint in merged_dataset.datapoint_list:
            if datapoint.klass not in sorted_datapoints.keys(): sorted_datapoints.update({klass: []})
            sorted_datapoints[klass].append(datapoint)

        for klass, datapoints in sorted_datapoints.items():
            mapped_datasets.add(
                Dataset(klass=klass, branches=self.branches, datapoint_list=datapoints))

        # TODO for batch/iterative run use here files from disk
        split_datasets = set()
        for dataset in mapped_datasets:
            # Reduce size of selected klasses
            if self.reducelist and dataset.klass in self.reducelist:
                logger.debug(f'Reducing number of samples in klass {klass}...')
                ratio = self.reduceratio[self.reducelist.index(klass)]
                dataset = dataset.reduce(ratio, seed=self.reduceseed)

            # Split datasets into train, validation, test and blackbox datasets
            if self.split == 'by_chr':
                split_subdatasets = Dataset.split_by_chr(dataset, self.chromosomes)
            elif self.split == 'rand':
                split_subdatasets = Dataset.split_random(dataset, self.splitratio_list, self.split_seed)
            split_datasets = split_datasets.union(split_subdatasets)

        # Merge datasets of the same category across all the branches (e.g. train = pos + neg)
        logger.debug('Merging dataset by category into final files and exporting...')
        final_datasets = Dataset.merge_by_category(split_datasets)

        for dataset in final_datasets:
            dir_path = os.path.join(self.output_folder, 'datasets', 'final_datasets')
            self.ensure_dir(dir_path)
            file_path = os.path.join(dir_path, dataset.category)
            dataset.save_to_file(file_path, zip=True)

        return final_datasets
