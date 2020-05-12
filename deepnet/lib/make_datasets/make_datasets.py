import datetime
import logging
import os
import shutil
import streamlit as st

from ..utils.dataset import Dataset
from ..utils.exceptions import UserInputError
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand


logger = logging.getLogger('root')


# noinspection DuplicatedCode
class MakeDatasets(Subcommand):

    def __init__(self):
        self.validation_hash = {'is_bed': [],
                                'is_fasta': [],
                                'is_wig_dir': [],
                                'not_empty_branches': [],
                                'min_one_file': [],
                                'is_full_dataset': [],
                                'is_ratio': [],
                                'not_empty_chromosomes': []}
        self.valid_chromosomes = []
        self.klasses = []

        st.markdown('# Data Preprocessing')

        # TODO add show/hide separate section after stateful operations are allowed
        st.markdown('## General Options')
        self.add_general_options()

        self.use_mapped = st.checkbox('Use already mapped file from a previous run')

        if not self.use_mapped:
            self.references = {}

            if 'seq' in self.branches:
                alphabets = {'DNA': ['A', 'C', 'G', 'T', 'N'],
                             'RNA': ['A', 'C', 'G', 'U', 'N']}
                # TODO allow option custom, to be specified by text input
                # TODO add amino acid alphabet - in that case disable cons and fold i guess
                self.onehot = alphabets[st.selectbox(
                    'Select used alphabet:',
                    list(alphabets.keys())
                )]
                self.strand = st.checkbox('Apply strandedness', True)
            else:
                self.onehot = None; self.strand = False
            if 'seq' in self.branches or 'fold' in self.branches:
                fasta = st.text_input('Path to reference fasta file')
                self.references.update({'seq': fasta, 'fold': fasta})
                self.validation_hash['is_fasta'].append(fasta)
            if 'cons' in self.branches:
                consdir = st.text_input('Path to folder containing reference conservation files')
                self.references.update({'cons': consdir})
                self.validation_hash['is_wig_dir'].append(consdir)

            self.win = int(st.number_input('Window size', min_value=3, max_value=500, value=100))
            self.winseed = int(st.number_input('Seed for semi-random window placement upon the sequences', value=42))

            st.markdown('## Input Coordinate Files')
            # TODO change to plus button when stateful operations enabled
            # TODO is there streamlit function to browse local files - should be soon
            # TODO accept also web address if possible

            warning = st.empty()
            self.input_files = []
            no_files = st.number_input('Number of input files (= no. of classes):', min_value=1, value=1)
            for i in range(no_files):
                self.input_files.append(st.text_input(f'File no. {i+1} (.bed)'))
            self.validation_hash['min_one_file'].append(list(filter(str.strip, self.input_files)))

            self.allowed_extensions = ['.bed', '.narrowPeak']
            for file in self.input_files:
                if not file: continue
                self.validation_hash['is_bed'].append(file)
                file_name = os.path.basename(file)
                if any(ext in file_name for ext in self.allowed_extensions):
                    for ext in self.allowed_extensions:
                        if ext in file_name:
                            self.klasses.append(file_name.replace(ext, ''))
                else:
                    warning.markdown(
                        '**WARNING: Only files of following format are allowed: {}.**'.format(', '.join(self.allowed_extensions)))
        else:
            # When using already mapped file
            self.full_dataset_file = st.text_input(f'Path to the mapped file')
            self.validation_hash['is_full_dataset'].append({'file_path': self.full_dataset_file, 'branches': self.branches})

            if self.full_dataset_file:
                try:
                    self.klasses, self.valid_chromosomes = Dataset.load_and_cache(self.full_dataset_file)
                except Exception:
                    raise UserInputError('Invalid dataset file!')

        st.markdown('## Dataset Size Reduction')
        st.markdown('Warning: the data are reduced randomly across the dataset. Thus in a rare occasion, when later '
                    'splitting the dataset by chromosomes, some categories might end up empty.')
        self.reducelist = st.multiselect('Classes to be reduced (first specify input files)', self.klasses)
        if self.reducelist:
            self.reduceseed = int(st.number_input('Seed for semi-random reduction of number of samples', value=112))
            self.reduceratio = {}
            for klass in self.reducelist:
                self.reduceratio.update({klass: float(st.number_input("Target {} dataset size".format(klass),
                                                                      min_value=0.01, max_value=1.0, value=0.5))})

        st.markdown('## Data Split')
        split_options = {'Random': 'rand',
                         'By chromosomes': 'by_chr'}
        self.split = split_options[st.radio(
            # 'Choose a way to split Datasets into train, test, validation and blackbox categories:',
            'Choose a way to split Datasets into train, test and validation categories:',
            list(split_options.keys()))]
        if self.split == 'by_chr':
            if self.use_mapped:
                if not self.full_dataset_file:
                    st.markdown('(The mapped file must be provided first to infer available chromosomes.)')
            else:
                if 'seq' in self.references.keys():
                    fasta = self.references['seq']
                    if not fasta:
                        st.markdown('(Fasta file with reference genome must be provided to infer available chromosomes.)')
                else:
                    st.markdown('**Please fill in a path to the fasta file below.** (Or you can specify it above if you select sequence or structure branch.)')
                    fasta = st.text_input('Path to reference fasta file')

                if fasta:
                    try:
                        fasta_dict, self.valid_chromosomes = seq.read_and_cache(fasta)
                    except Exception:
                        raise UserInputError('Sorry, could not parse given fasta file. Please check the path.')
                    self.references.update({'seq': fasta_dict, 'fold': fasta_dict})

            if self.valid_chromosomes:
                if not self.use_mapped:
                    st.markdown("Note: While selecting the chromosomes, you may ignore the yellow warning box, \
                    and continue selecting even while it's present.")

                self.chromosomes = {'blackbox': []}
                self.chromosomes.update({'train':
                                             set(st.multiselect('Training Dataset', self.valid_chromosomes, None))})
                self.chromosomes.update({'validation':
                                             set(st.multiselect('Validation Dataset', self.valid_chromosomes, None))})
                self.chromosomes.update({'test':
                                             set(st.multiselect('Test Dataset', self.valid_chromosomes, None))})
                # self.chromosomes.update({'blackbox':
                #                              set(st.multiselect('BlackBox Dataset', self.valid_chromosomes, None))})
                self.validation_hash['not_empty_chromosomes'].append(list(self.chromosomes.items()))
        elif self.split == 'rand':
            self.split_ratio = st.text_input(
                # 'List a target ratio between the categories (required format: train:validation:test:blackbox)',
                'List a target ratio between the categories (required format: train:validation:test)',
                value='8:2:2')
            self.validation_hash['is_ratio'].append(self.split_ratio)
            # st.markdown('Note: blackbox dataset usage is currently not yet implemented, thus recommended value is 0.')
            self.split_seed = int(st.number_input('Seed for semi-random split of samples in a Dataset', value=89))

        self.validate_and_run(self.validation_hash)

    def run(self):
        status = st.empty()

        datasets_dir = os.path.join(self.output_folder, 'datasets', f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))}')
        self.ensure_dir(datasets_dir)
        full_data_dir_path = os.path.join(datasets_dir, 'full_datasets')
        self.ensure_dir(full_data_dir_path)
        full_data_file_path = os.path.join(full_data_dir_path, 'merged_all')

        if self.use_mapped:
            status.text('Reading in already mapped file with all the samples...')
            merged_dataset = Dataset.load_from_file(self.full_dataset_file)
            shutil.copyfile(self.full_dataset_file, full_data_file_path)
            # Keep only selected branches
            cols = ['chrom_name', 'seq_start', 'seq_end', 'strand_sign', 'klass'] + self.branches
            merged_dataset.df = merged_dataset.df[cols]
        else:
            if self.onehot:
                status.text('Encoding alphabet...')
                encoding = seq.onehot_encode_alphabet(self.onehot)
            else:
                encoding = None

            # Accept one file per class and create one Dataset per each
            initial_datasets = set()
            status.text('Reading in given interval files and applying window...')
            for file in self.input_files:
                klass = os.path.basename(file)
                for ext in self.allowed_extensions:
                    if ext in klass:
                        klass = klass.replace(ext, '')

                initial_datasets.add(
                    Dataset(klass=klass, branches=self.branches, bed_file=file, win=self.win, winseed=self.winseed))

            # Merging data from all klasses to map them more efficiently all together at once
            merged_dataset = Dataset(branches=self.branches, df=Dataset.merge_dataframes(initial_datasets))

            # Read-in fasta file to dictionary
            if 'seq' in self.branches and type(self.references['seq']) != dict:
                status.text('Reading in reference fasta file...')
                fasta_dict, _ = seq.parse_fasta_reference(self.references['seq'])
                self.references.update({'seq': fasta_dict, 'fold': fasta_dict})

            # First ensure order of the data by chr_name and seq_start within, mainly for conservation
            status.text(
                f'Mapping intervals from all classes to {len(self.branches)} branch(es) and exporting...')
            merged_dataset.sort_datapoints().map_to_branches(
                self.references, encoding, self.strand, full_data_file_path, self.ncpu)

        status.text('Processing mapped samples...')
        mapped_datasets = set()
        for klass in self.klasses:
            df = merged_dataset.df[merged_dataset.df['klass'] == klass]
            mapped_datasets.add(Dataset(klass=klass, branches=self.branches, df=df))

        split_datasets = set()
        for dataset in mapped_datasets:
            # Reduce size of selected klasses
            if self.reducelist and (dataset.klass in self.reducelist):
                status.text(f'Reducing number of samples in klass {format(dataset.klass)}...')
                ratio = self.reduceratio[dataset.klass]
                dataset.reduce(ratio, seed=self.reduceseed)

            # Split datasets into train, validation, test and blackbox datasets
            if self.split == 'by_chr':
                split_subdatasets = Dataset.split_by_chr(dataset, self.chromosomes)
            elif self.split == 'rand':
                split_subdatasets = Dataset.split_random(dataset, self.split_ratio, self.split_seed)
            split_datasets = split_datasets.union(split_subdatasets)

        # Merge datasets of the same category across all the branches (e.g. train = pos + neg)
        status.text('Redistributing samples to categories and exporting into final files...')
        final_datasets = Dataset.merge_by_category(split_datasets)

        for dataset in final_datasets:
            dir_path = os.path.join(datasets_dir, 'final_datasets')
            self.ensure_dir(dir_path)
            file_path = os.path.join(dir_path, dataset.category)
            dataset.save_to_file(file_path, do_zip=True)

        self.finalize_run(logger, datasets_dir)
        status.text('Finished!')
        return final_datasets
