import datetime
import logging
import os
import streamlit as st

from ..utils.dataset import Dataset
from ..utils import file_utils as f
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand

logger = logging.getLogger('main')


# noinspection DuplicatedCode
class MakeDatasets(Subcommand):

    def __init__(self):
        st.header('Data Preprocessing')

        # TODO add show/hide separate section after stateful operations are allowed
        st.subheader('General Options')
        self.add_general_options()

        self.references = {}

        if 'seq' in self.branches:
            alphabets = {'DNA': ['A', 'C', 'G', 'T', 'N'],
                         'RNA': ['A', 'C', 'G', 'U', 'N']}
            # TODO allow option custom, to be specified by text input
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
        if 'cons' in self.branches:
            consdir = st.text_input('Path to folder containing reference conservation files')
            self.references.update({'cons': consdir})

        self.win = int(st.number_input('Window size', min_value=0, max_value=500, value=100))
        self.winseed = int(st.number_input('Seed for semi-random window placement upon the sequences', value=42))

        st.subheader('Input Coordinate Files')
        # TODO change to plus button when stateful operations enabled
        # TODO is there streamlit function to browse local files - should be soon
        # TODO accept also web address - also should be soon possible

        warning = st.empty()
        self.input_files = []
        no_files = st.number_input('Number of input files (= no. of classes):', min_value=0)
        for i in range(no_files):
            self.input_files.append(st.text_input(f'File no. {i+1} (.bed or .narrowPeak)'))

        self.klasses = []
        self.allowed_extensions = ['.bed', '.narrowPeak']
        for file in self.input_files:
            file_name = os.path.basename(file)
            if any(ext in file_name for ext in self.allowed_extensions):
                for ext in self.allowed_extensions:
                    if ext in file_name:
                        self.klasses.append(file_name.replace(ext, ''))
            else:
                warning.text(
                    'WARNING: Only files of following format are allowed: {}.'.format(', '.join(self.allowed_extensions)))

        st.subheader('Data Split')
        split_options = {'By chromosomes': 'by_chr',
                         'Random': 'rand'}
        self.split = split_options[st.radio(
            'Choose a way to split Datasets into train, test, validation and blackbox categories:',
            list(split_options.keys()))]
        if self.split == 'by_chr':
            self.chromosomes = {}
            self.chromosomes.update({'validation':
                                         set(st.multiselect('Validation Dataset', seq.VALID_CHRS, {'chr19', 'chr20'}))})
            self.chromosomes.update({'test':
                                         set(st.multiselect('Test Dataset', seq.VALID_CHRS, {'chr21'}))})
            self.chromosomes.update({'blackbox':
                                         set(st.multiselect('BlackBox Dataset', seq.VALID_CHRS, {'chr22'}))})
            default_train_set = (set(seq.VALID_CHRS) - self.chromosomes['validation'] - self.chromosomes['test'] -
                                 self.chromosomes['blackbox'])
            self.chromosomes.update({'train':
                                         set(st.multiselect('Training Dataset', seq.VALID_CHRS, default_train_set))})
        elif self.split == 'rand':
            self.splitratio_list = st.text_input(
                'List a target ratio between the categories (required format: train:validation:test:blackbox)',
                value='8:2:2:1').split(':')
            self.split_seed = int(st.number_input('Seed for semi-random split of samples in a Dataset', value=89))

        st.subheader('Dataset Size Reduction')
        self.reducelist = st.multiselect('Klasses to be reduced (first specify input files)', self.klasses)
        if self.reducelist:
            self.reduceseed = int(st.number_input('Seed for semi-random reduction of number of samples', value=112))
            self.reduceratio = {}
            for klass in self.reducelist:
                self.reduceratio.update({klass: float(st.number_input("Target {} dataset size".format(klass),
                                                                      min_value=0.0, max_value=1.0, value=0.5))})

        st.markdown('---')
        if st.button('Run preprocessing'):
            # TODO check input presence & validity, if OK continue to run
            self.run()

    def run(self):
        status = st.empty()

        datasets_dir = os.path.join(self.output_folder, 'datasets', f'{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        self.ensure_dir(datasets_dir)

        zipped_files = f.list_files_in_dir(datasets_dir, 'zip')
        result = [file for file in zipped_files if 'full_datasets/merged_all.zip' in file]

        if len(result) == 1:
            status.text('Reading in already mapped file with all the samples...')
            merged_dataset = Dataset.load_from_file(result[0])
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

            # Merging datapoints from all klasses to map them more quickly all together at once
            all_datapoints = []
            for dataset in initial_datasets:
                all_datapoints += dataset.datapoint_list
            merged_dataset = Dataset(branches=self.branches, datapoint_list=all_datapoints)

            # Read-in fasta file to dictionary
            if 'seq' in self.branches or 'fold' in self.branches:
                status.text('Reading in reference fasta file...')
                fasta_dict = seq.fasta_to_dictionary(self.references['seq'])
                self.references.update({'seq': fasta_dict, 'fold': fasta_dict})

            dir_path = os.path.join(datasets_dir, 'full_datasets')
            self.ensure_dir(dir_path)
            outfile_path = os.path.join(dir_path, 'merged_all')

            # First ensure order of the DataPoints by chr_name and seq_start within, mainly for conservation
            status.text(
                f'Mapping intervals from all classes to {len(self.branches)} branch(es) and exporting...')
            merged_dataset.sort_datapoints().map_to_branches(
                self.references, encoding, self.strand, outfile_path, self.ncpu)

        status.text('Processing mapped samples...')
        mapped_datasets = set()
        sorted_datapoints = {}
        for datapoint in merged_dataset.datapoint_list:
            if datapoint.klass not in sorted_datapoints.keys(): sorted_datapoints.update({datapoint.klass: []})
            sorted_datapoints[datapoint.klass].append(datapoint)

        for klass, datapoints in sorted_datapoints.items():
            mapped_datasets.add(
                Dataset(klass=klass, branches=self.branches, datapoint_list=datapoints))

        split_datasets = set()
        for dataset in mapped_datasets:
            # Reduce size of selected klasses
            if self.reducelist and dataset.klass in self.reducelist:
                status.text('Reducing number of samples in klass {}...'.format(dataset.klass))
                ratio = self.reduceratio[dataset.klass]
                dataset = dataset.reduce(ratio, seed=self.reduceseed)

            # Split datasets into train, validation, test and blackbox datasets
            if self.split == 'by_chr':
                split_subdatasets = Dataset.split_by_chr(dataset, self.chromosomes)
            elif self.split == 'rand':
                split_subdatasets = Dataset.split_random(dataset, self.splitratio_list, self.split_seed)
            split_datasets = split_datasets.union(split_subdatasets)

        # Merge datasets of the same category across all the branches (e.g. train = pos + neg)
        status.text('Redistributing samples to categories and exporting into final files...')
        final_datasets = Dataset.merge_by_category(split_datasets)

        for dataset in final_datasets:
            dir_path = os.path.join(datasets_dir, 'final_datasets')
            self.ensure_dir(dir_path)
            file_path = os.path.join(dir_path, dataset.category)
            dataset.save_to_file(file_path, zip=True)

        status.text('Finished!')
        return final_datasets
