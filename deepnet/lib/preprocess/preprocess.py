import datetime
import logging
import os
import shutil
import streamlit as st
import subprocess

from ..utils.dataset import Dataset
from ..utils.exceptions import UserInputError
from ..utils import sequence as seq
from ..utils.subcommand import Subcommand


logger = logging.getLogger('root')


# noinspection DuplicatedCode
class Preprocess(Subcommand):
    ALPHABETS = {'DNA': ['A', 'C', 'G', 'T', 'N'],
                 'RNA': ['A', 'C', 'G', 'U', 'N']}

    def __init__(self):
        self.params = {'task': 'Preprocess'}
        self.validation_hash = {'is_bed': [],
                                'is_fasta': [],
                                'is_wig_dir': [],
                                'not_empty_branches': [],
                                'min_two_files': [],
                                'is_full_dataset': [],
                                'is_ratio': [],
                                'not_empty_chromosomes': []}
        self.klasses = []
        self.klass_sizes = {}

        st.markdown('# Data Preprocessing')

        # TODO add show/hide separate section after stateful operations are allowed
        st.markdown('## General Options')
        self.add_general_options()

        self.params['use_mapped'] = st.checkbox('Use already mapped file from a previous run', self.defaults['use_mapped'])

        if not self.params['use_mapped']:
            self.references = {}
            if 'seq' in self.params['branches']:
                # TODO allow option custom, to be specified by text input
                # TODO add amino acid alphabet - in that case disable cons and fold i guess
                alphabets = ['DNA', 'RNA']
                self.params['alphabet'] = st.selectbox('Select alphabet:',
                                                       alphabets, index=alphabets.index(self.defaults['alphabet']))
                self.params['strand'] = st.checkbox('Apply strandedness', self.defaults['strand'])
            if 'seq' in self.params['branches'] or 'fold' in self.params['branches']:
                self.params['fasta'] = st.text_input('Path to reference fasta file', value=self.defaults['fasta'])
                self.references.update({'seq': self.params['fasta'], 'fold': self.params['fasta']})
                self.validation_hash['is_fasta'].append(self.params['fasta'])
            if 'cons' in self.params['branches']:
                self.params['cons_dir'] = st.text_input('Path to folder containing reference conservation files',
                                                        value=self.defaults['cons_dir'])
                self.references.update({'cons': self.params['cons_dir']})
                self.validation_hash['is_wig_dir'].append(self.params['cons_dir'])

            self.params['win'] = int(st.number_input('Window size', min_value=3, value=self.defaults['win']))
            self.params['winseed'] = int(st.number_input('Seed for semi-random window placement upon the sequences',
                                                         value=self.defaults['winseed']))

            st.markdown('## Input Coordinate Files')
            # TODO change to plus button when stateful operations enabled
            # TODO is there streamlit function to browse local files - should be soon
            # TODO accept also web address if possible

            warning = st.empty()
            self.params['input_files'] = self.defaults['input_files']
            no_files = st.number_input('Number of input files (= no. of classes):', min_value=2,
                                       value=max(2, len(self.defaults['input_files'])))
            for i in range(no_files):
                self.params['input_files'].append(st.text_input(
                    f'File no. {i+1} (.bed)',
                    value=(self.defaults['input_files'][i] if len(self.defaults['input_files']) > i else '')))
            self.validation_hash['min_two_files'].append(list(filter(str.strip, self.params['input_files'])))

            self.allowed_extensions = ['.bed', '.narrowPeak']
            for file in self.params['input_files']:
                if not file: continue
                self.validation_hash['is_bed'].append(file)
                file_name = os.path.basename(file)
                if any(ext in file_name for ext in self.allowed_extensions):
                    for ext in self.allowed_extensions:
                        if ext in file_name:
                            klass = file_name.replace(ext, '')
                            self.klasses.append(klass)
                            # subprocess.run(['wc', '-l', file], check=True)
                            self.klass_sizes.update({klass: (int(subprocess.check_output(['wc', '-l', file]).split()[0]))})
                else:
                    warning.markdown(
                        '**WARNING**: Only files of following format are allowed: {}.'.format(', '.join(self.allowed_extensions)))
        else:
            # When using already mapped file
            self.params['full_dataset_file'] = st.text_input(f'Path to the mapped file', value=self.defaults['full_dataset_file'])
            self.validation_hash['is_full_dataset'].append({'file_path': self.params['full_dataset_file'], 'branches': self.params['branches']})

            if self.params['full_dataset_file']:
                try:
                    self.klasses, self.params['valid_chromosomes'] = Dataset.load_and_cache(self.params['full_dataset_file'])
                except Exception:
                    raise UserInputError('Invalid dataset file!')

        st.markdown('## Dataset Size Reduction')
        st.markdown('Input a decimal number if you want to reduce the sample size by a ratio (e.g. 0.1 to get 10%),'
                    'or an integer if you wish to select final dataset size (e.g. 5000 if you want exactly 5000 samples).')
        self.params['reducelist'] = st.multiselect('Classes to be reduced (first specify input files)',
                                                   self.klasses, self.defaults['reducelist'])
        if self.params['reducelist']:
            self.params['reduceseed'] = int(st.number_input('Seed for semi-random reduction of number of samples',
                                                            value=self.defaults['reduceseed']))
            self.params['reduceratio'] = self.defaults['reduceratio']
            for klass in self.params['reducelist']:
                self.params['reduceratio'].update({klass: float(st.number_input(
                    f'Target {klass} dataset size (original size: {self.klass_sizes[klass]} rows)',
                    min_value=0.00001, value=0.01, format='%.5f'))})
        st.markdown('###### Warning: the data are reduced randomly across the dataset. Thus in a rare occasion, when later '
                    'splitting the dataset by chromosomes, some categories might end up empty. Thus it\'s recommended '
                    'to be used in combination with random split.')

        st.markdown('## Data Split')
        split_options = {'Random': 'rand',
                         'By chromosomes': 'by_chr'}
        self.params['split'] = split_options[st.radio(
            # 'Choose a way to split Datasets into train, test, validation and blackbox categories:',
            'Choose a way to split Datasets into train, test and validation categories:',
            list(split_options.keys()), index=self.get_dict_index(self.defaults['split'], split_options))]
        if self.params['split'] == 'by_chr':
            if self.params['use_mapped']:
                if not self.params['full_dataset_file']:
                    st.markdown('(The mapped file must be provided first to infer available chromosomes.)')
            else:
                if 'seq' in self.references.keys():
                    self.params['fasta'] = self.references['seq']
                    if not self.params['fasta']:
                        st.markdown('(Fasta file with reference genome must be provided to infer available chromosomes.)')
                else:
                    st.markdown('**Please fill in a path to the fasta file below.** (Or you can specify it above if you select sequence or structure branch.)')
                    self.params['fasta'] = st.text_input('Path to reference fasta file')

                if self.params['fasta']:
                    try:
                        fasta_dict, self.params['valid_chromosomes'] = seq.read_and_cache(self.params['fasta'])
                    except Exception:
                        raise UserInputError('Sorry, could not parse given fasta file. Please check the path.')
                    self.references.update({'seq': fasta_dict, 'fold': fasta_dict})

            print(self.params['valid_chromosomes'])
            if self.params['fasta']:
                if self.params['valid_chromosomes']:
                    if not self.params['use_mapped']:
                        st.markdown("Note: While selecting the chromosomes, you may ignore the yellow warning box, \
                        and continue selecting even while it's present.")
                    self.params['chromosomes'] = self.defaults['chromosomes']
                    self.params['chromosomes'].update({'train': set(st.multiselect(
                        'Training Dataset', self.params['valid_chromosomes'], list(self.defaults['chromosomes']['train'])))})
                    self.params['chromosomes'].update({'validation': set(
                       st.multiselect('Validation Dataset', self.params['valid_chromosomes'], list(self.defaults['chromosomes']['validation'])))})
                    self.params['chromosomes'].update({'test': set(
                        st.multiselect('Test Dataset', self.params['valid_chromosomes'], list(self.defaults['chromosomes']['test'])))})
                    # self.params['chromosomes'].update({'blackbox': set(
                    #   st.multiselect('BlackBox Dataset', self.params['valid_chromosomes'], list(self.defaults['chromosomes']['blackbox'])))})
                    self.validation_hash['not_empty_chromosomes'].append(list(self.params['chromosomes'].items()))
                else:
                    raise UserInputError('Sorry, did not find any valid chromosomes in given fasta file.')

        elif self.params['split'] == 'rand':
            self.params['split_ratio'] = st.text_input(
                # 'List a target ratio between the categories (required format: train:validation:test:blackbox)',
                'List a target ratio between the categories (required format: train:validation:test)',
                value=self.defaults['split_ratio'])
            self.validation_hash['is_ratio'].append(self.params['split_ratio'])
            # st.markdown('Note: blackbox dataset usage is currently not yet implemented, thus recommended value is 0.')
            self.params['split_seed'] = int(st.number_input('Seed for semi-random split of samples in a Dataset',
                                                            value=self.defaults['split_seed']))

        self.validate_and_run(self.validation_hash)

    def run(self):
        status = st.empty()

        datasets_dir = os.path.join(self.params['output_folder'], 'datasets', f'{str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"))}')
        self.ensure_dir(datasets_dir)
        full_data_dir_path = os.path.join(datasets_dir, 'full_datasets')
        self.ensure_dir(full_data_dir_path)
        full_data_file_path = os.path.join(full_data_dir_path, 'merged_all')

        if self.params['use_mapped']:
            status.text('Reading in already mapped file with all the samples...')
            merged_dataset = Dataset.load_from_file(self.params['full_dataset_file'])
            shutil.copyfile(self.params['full_dataset_file'], full_data_file_path)
            # Keep only selected branches
            cols = ['chrom_name', 'seq_start', 'seq_end', 'strand_sign', 'klass'] + self.params['branches']
            merged_dataset.df = merged_dataset.df[cols]
        else:
            if self.params['alphabet']:
                status.text('Encoding alphabet...')
                encoding = seq.onehot_encode_alphabet(self.ALPHABETS[self.params['alphabet']])
            else:
                encoding = None

            # Accept one file per class and create one Dataset per each
            initial_datasets = set()
            status.text('Reading in given interval files and applying window...')
            for file in self.params['input_files']:
                klass = os.path.basename(file)
                for ext in self.allowed_extensions:
                    if ext in klass:
                        klass = klass.replace(ext, '')

                initial_datasets.add(
                    Dataset(klass=klass, branches=self.params['branches'], bed_file=file, win=self.params['win'], winseed=self.params['winseed']))

            # Merging data from all klasses to map them more efficiently all together at once
            merged_dataset = Dataset(branches=self.params['branches'], df=Dataset.merge_dataframes(initial_datasets))

            # Read-in fasta file to dictionary
            if 'seq' in self.params['branches'] and type(self.references['seq']) != dict:
                status.text('Reading in reference fasta file...')
                fasta_dict, valid_chromosomes = seq.parse_fasta_reference(self.references['seq'])
                if not valid_chromosomes:
                    raise UserInputError('Sorry, did not find any valid chromosomes in given fasta file.')
                self.references.update({'seq': fasta_dict, 'fold': fasta_dict})

            # First ensure order of the data by chr_name and seq_start within, mainly for conservation
            status.text(
                f"Mapping intervals from all classes to {len(self.params['branches'])} branch(es) and exporting...")
            merged_dataset.sort_datapoints().map_to_branches(
                self.references, encoding, self.params['strand'], full_data_file_path, self.ncpu)

        status.text('Processing mapped samples...')
        mapped_datasets = set()
        for klass in self.klasses:
            df = merged_dataset.df[merged_dataset.df['klass'] == klass]
            mapped_datasets.add(Dataset(klass=klass, branches=self.params['branches'], df=df))

        split_datasets = set()
        for dataset in mapped_datasets:
            # Reduce size of selected klasses
            if self.params['reducelist'] and (dataset.klass in self.params['reducelist']):
                status.text(f'Reducing number of samples in klass {format(dataset.klass)}...')
                ratio = self.params['reduceratio'][dataset.klass]
                dataset.reduce(ratio, seed=self.params['reduceseed'])

            # Split datasets into train, validation, test and blackbox datasets
            if self.params['split'] == 'by_chr':
                split_subdatasets = Dataset.split_by_chr(dataset, self.params['chromosomes'])
            elif self.params['split'] == 'rand':
                split_subdatasets = Dataset.split_random(dataset, self.params['split_ratio'], self.params['split_seed'])
            split_datasets = split_datasets.union(split_subdatasets)

        # Merge datasets of the same category across all the branches (e.g. train = pos + neg)
        status.text('Redistributing samples to categories and exporting into final files...')
        final_datasets = Dataset.merge_by_category(split_datasets)

        for dataset in final_datasets:
            dir_path = os.path.join(datasets_dir, 'final_datasets')
            self.ensure_dir(dir_path)
            file_path = os.path.join(dir_path, dataset.category)
            dataset.save_to_file(file_path, do_zip=True)

        self.finalize_run(logger, datasets_dir, self.params, self.csv_header(), self.csv_row(datasets_dir, self.params))
        status.text('Finished!')

    @staticmethod
    def default_params():
        return {'alphabet': 'DNA',
                'branches': [],
                'chromosomes': {'train': [], 'validation': [], 'test': [], 'blackbox': []},
                'cons_dir': '',
                'fasta': '',
                'full_dataset_file': '',
                'input_files': [],
                'output_folder': os.path.join(os.getcwd(), 'deepnet_output'),
                'reducelist': [],
                'reduceratio': {},
                'reduceseed': 112,
                'split': 'rand',
                'split_ratio': '8:2:2',
                'split_seed': 89,
                'strand': True,
                'use_mapped': False,
                'valid_chromosomes': [],
                'win': 100,
                'winseed': 42
                }

    @staticmethod
    def csv_header():
        return 'Folder\t' \
               'Branches\t' \
               'Alphabet\t' \
               'Strand\t' \
               'Window\t' \
               'Window seed\t' \
               'Split\t' \
               'Split ratio\t' \
               'Split seed\t' \
               'Chromosomes\t' \
               'Reduced classes\t' \
               'Reduce ratio\t' \
               'Reduce seed\t' \
               'Use mapped\t' \
               'Input files\t' \
               'Full_dataset_file\t' \
               'Fasta ref\t' \
               'Conservation ref\n'

    @staticmethod
    def csv_row(folder, params):
        return f"{os.path.basename(folder)}\t" \
               f"{[Preprocess.get_dict_key(b, Preprocess.BRANCHES) for b in params['branches']]}\t" \
               f"{params['alphabet'] if 'seq' in params['branches'] else '-'}\t" \
               f"{'Yes' if (params['strand'] and 'seq' in params['branches']) else ('No' if 'seq' in params['branches'] else '-')}\t" \
               f"{params['win']}\t" \
               f"{params['winseed']}\t" \
               f"{'Random' if params['split'] == 'rand' else 'By chromosomes'}\t" \
               f"{params['split_ratio'] if params['split'] == 'rand' else '-'}\t" \
               f"{params['split_seed'] if params['split'] == 'rand' else '-'}\t" \
               f"{params['chromosomes'] if params['split'] == 'by_chr' else '-'}\t" \
               f"{params['reducelist'] if len(params['reducelist']) != 0 else '-'}\t" \
               f"{params['reduceratio'] if len(params['reducelist']) != 0 else '-'}\t" \
               f"{params['reduceseed'] if len(params['reducelist']) != 0 else '-'}\t" \
               f"{'Yes' if params['use_mapped'] else 'No'}\t" \
               f"{params['input_files']}\t" \
               f"{params['full_dataset_file'] if params['use_mapped'] else '-'}\t" \
               f"{params['fasta'] if params['fasta'] else '-'}\t" \
               f"{params['cons_dir'] if params['cons_dir'] else '-'}\n"
