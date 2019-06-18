import os
import sys
import logging

from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

from .dataset import Dataset

logger = logging.getLogger('main')

class MakeDatasets(Subcommand):

    def __init__(self):
        help_message = '''deepnet <subcommand> [<args>]
            Preprocess input files by creating datasets containing data specific per each branch. That includes mapping
            to reference, encoding, folding, etc.    
            
            Validation, test and blackbox datasets can be assigned specific chromosomes. The rest of the chromosomes 
            is used for training.
            '''

        parser = self.initialize_parser(help_message)

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
            help="Branches. [default: %default]"
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
            help="If data needs to be converted to One Hot encoding, give a list of alphabet used.",
            dest="onehot",
            default=None
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
        logger.info('Running make_datasets with the following arguments: ' + str(self.args)[10:-1])

        self.encoded_alphabet = None
        self.seq_ref = seq.fasta_to_dictionary(self.args.ref)
        self.branches = self.args.branches
        if 'cons' in self.branches:
            if not self.args.consdir:
                logger.exception('Exception occured.')
                raise Exception("Provide conservation directory for calculating scores for conservation branch.")
            else:
                self.cons_ref = seq.wig_to_dictionary(self.args.consdir)

        if self.args.coord:
            self.input_files = self.args.coord
        else:
            logger.exception('Exception occured.')
            raise Exception("Input coordinate (.bed) files are required. Provide one file per class.")

        self.chromosomes = {'validation': self.args.validation,
                            'test': self.args.test,
                            'blackbox': self.args.blackbox,
                            'train': (seq.VALID_CHRS - self.args.validation - self.args.test - self.args.blackbox)}
        # TODO add more printouts for verbose throughout the code
        if self.args.verbose:
            logger.info('Running deepnet make_datasets with input files {}'.format(', '.join(self.input_files)))

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
            datasets.update({branch: Dataset(branch, datasetlist=datasets[branch].values())})

        # Separate data into train, validation, test and blackbox datasets
        separated_datasets = {}
        for branch in datasets.keys():
            for category in self.chromosomes.keys():
                key = "{}_{}".format(branch, category)
                value = datasets[branch].separate_by_chr(self.chromosomes[category])
                separated_datasets.update({key: value})

        # Final datasets in format branch_category, e.g. 'seq_test' or 'fold_train'
        return separated_datasets
