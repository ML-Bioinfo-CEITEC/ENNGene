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

        logger.debug('Converting reference fasta file into dictionary.')
        time1 = datetime.now()
        seq_dictionary = seq.fasta_to_dictionary(self.args.ref)
        logger.debug('Finished converting reference file in {}.'.format(self.spent_time(time1)))
        self.reference_files = {'seq': seq_dictionary, 'fold': seq_dictionary}

        self.branches = self.args.branches
        if 'cons' in self.branches:
            if not self.args.consdir:
                logger.exception('Exception occurred.')
                raise Exception("Provide conservation directory for calculating scores for conservation branch.")
            else:
                logger.debug('Converting reference wig files into dictionary.')
                time1 = datetime.now()
                self.reference_files.update({'cons': seq.wig_to_dictionary(self.args.consdir)})
                logger.debug('Finished converting reference files in {}.'.format(self.spent_time(time1)))

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

        logger.debug('Finished initialization.')
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

    def run(self):
        if self.args.onehot:
            logger.debug('Encoding alphabet.')
            time1 = datetime.now()
            encoding = seq.onehot_encode_alphabet(self.args.onehot)
            logger.debug('Finished encoding alphabet in {}.'.format(self.spent_time(time1)))
        else:
            encoding = None

        # Accept one file per class and create one Dataset per each
        full_datasets = set()
        logger.debug('Preparing initial datasets per each input file.')
        time1 = datetime.now()
        for file in self.input_files:
            file_name = os.path.basename(file)
            if '.bed' in file_name:
                klass = file_name.replace('.bed', '')
            else:
                klass = file_name

            logger.debug('Preparing initial dataset for klass {}.'.format(klass))
            sub_time1 = datetime.now()
            full_datasets.add(Dataset(klass=klass,
                                      branches=self.branches,
                                      bed_file=file,
                                      ref_files=self.reference_files,
                                      strand=self.args.strand,
                                      encoding=encoding))
            logger.debug('Finished creating the dataset in {}.'.format(self.spent_time(sub_time1)))
        logger.debug('Finished creating initial datasets in {}.'.format(self.spent_time(time1)))

        logger.debug('Writing initial datasets into files...')
        time1 = datetime.now()
        for dataset in full_datasets:
            dir_path = os.path.join(self.output_folder, 'datasets', 'full_datasets')
            file_name = dataset.klass + '_' + '_'.join(dataset.branches)
            logger.debug('Saving {} as {} to: {}...'.format(dataset.klass, file_name, dir_path))
            dataset.save_to_file(dir_path, file_name)
        logger.debug('Finished saving the files in {}.'.format(self.spent_time(time1)))

        # Reduce dataset for selected classes to lower the amount of samples in overpopulated classes
        if self.reducelist:
            logger.debug('Reducing selected datasets...')
            time1 = datetime.now()
            reduced_datasets = set()
            for dataset in full_datasets:
                if dataset.klass in self.reducelist:
                    logger.info("Reducing number of samples in klass {}...".format(dataset.klass))
                    ratio = self.reduceratio[self.reducelist.index(dataset.klass)]
                    reduced_datasets.add(dataset.reduce(ratio, self.reduceseed))
                    # Save the reduced datasets
                    dir_path = os.path.join(self.output_folder, 'datasets', 'reduced_datasets')
                    self.ensure_dir(dir_path)
                    file_name = dataset.klass + '_' + '_'.join(dataset.branches)
                    logger.debug('Saving {} as {} to: {}...'.format(dataset.klass, file_name, dir_path))
                    dataset.save_to_file(dir_path, file_name)
                else:
                    logger.info("Keeping full number of samples for klass {}".format(dataset.klass))
                    reduced_datasets.add(dataset)
            logger.debug('Finished reducing datasets in {}.'.format(self.spent_time(time1)))
        else:
            reduced_datasets = full_datasets

        # Split datasets into train, validation, test and blackbox datasets
        split_datasets = set()
        logger.debug('Splitting datasets into train, test, validation and blackbox categories...')
        time1 = datetime.now()
        for dataset in reduced_datasets:
            logger.debug('Splitting dataset for {}...'.format(dataset.klass))
            if self.split == 'by_chr':
                split_subdatasets = Dataset.split_by_chr(dataset, self.chromosomes)
            elif self.split == 'rand':
                split_subdatasets = Dataset.split_random(dataset, self.splitratio_list, self.split_seed)
            logger.debug('Finished splitting the dataset.')
            split_datasets = split_datasets.union(split_subdatasets)
        logger.debug('Finished splitting all the datasets in {}.'.format(self.spent_time(time1)))

        # Merge datasets of the same category across all the branches (e.g. train = pos + neg)
        logger.debug('Merging datasets of different klasses by categories...')
        time1 = datetime.now()
        final_datasets = Dataset.merge_by_category(split_datasets)
        logger.debug('Finished merging the datasets in {}.'.format(self.spent_time(time1)))

        # Export final datasets to files
        logger.debug('Writing final datasets into files...')
        time1 = datetime.now()
        for dataset in final_datasets:
            dir_path = os.path.join(self.output_folder, 'datasets', 'final_datasets')
            self.ensure_dir(dir_path)
            file_name = dataset.category
            logger.debug('Saving {} to: {}...'.format(dataset.category, dir_path))
            dataset.save_to_file(dir_path, file_name)
        logger.debug('Finished saving the files in {}.'.format(self.spent_time(time1)))

        logger.debug('Finished running make_datasets in {}.'.format(self.spent_time(self.total_time1)))
        return final_datasets
