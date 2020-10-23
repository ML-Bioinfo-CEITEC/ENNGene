import _io
import datetime
import logging
import numpy as np
import os
import pandas as pd
import streamlit as st
import subprocess
import sys
import tempfile

from functools import reduce
from zipfile import ZipFile, ZIP_DEFLATED

from .exceptions import UserInputError, ProcessError
from . import file_utils as f
from . import sequence as seq

logger = logging.getLogger('root')


class Dataset:

    @classmethod
    @st.cache(hash_funcs={_io.TextIOWrapper: lambda _: None}, suppress_st_warning=True)
    def load_and_cache(cls, file_path):
        with st.spinner('Reading in the mapped file. Might take up to few minutes...'):
            full_dataset = cls.load_from_file(file_path)
            klasses = list(full_dataset.df['klass'].unique())
            valid_chromosomes = list(full_dataset.df['chrom_name'].unique())
        return klasses, valid_chromosomes

    @classmethod
    def load_from_file(cls, file_path):
        name = os.path.basename(file_path).replace('.tsv.zip', '')
        df = pd.read_csv(file_path, sep='\t', header=0)
        branches = [col for col in df.columns if col not in ['chrom_name', 'seq_start', 'seq_end', 'strand_sign', 'klass', 'input']]
        category = name if (name in ['train', 'test', 'validation', 'blackbox']) else None

        return cls(branches=branches, category=category, df=df)

    @classmethod
    def split_by_chr(cls, dataset, chrs_by_category):
        split_datasets = set()
        for category, chr_list in chrs_by_category.items():
            df = dataset.df[dataset.df['chrom_name'].isin(chr_list)]
            split_datasets.add(
                Dataset(klass=dataset.klass, branches=dataset.branches, category=category, df=df))

        return split_datasets

    @classmethod
    def split_random(cls, dataset, ratio, seed):
        # so far the categories are fixed, not sure if there would be need for custom categories
        ratio_list = ratio.split(':')
        ratio_list = [float(x) for x in ratio_list]
        dataset_size = dataset.df.shape[0]
        total = sum(ratio_list)
        np.random.seed(seed)
        np.random.shuffle(dataset.df.values)
        split_datasets = set()

        validation_size = int(dataset_size * ratio_list[1] / total)
        test_size = (int(dataset_size * ratio_list[2] / total))
        blackbox_size = (int(dataset_size * ratio_list[3] / total))
        dfs = {'validation': dataset.df[:validation_size]}
        dfs.update({'test': dataset.df[validation_size:(validation_size + test_size)]})
        dfs.update({'blackbox': dataset.df[(validation_size + test_size):(validation_size + test_size + blackbox_size)]})
        dfs.update({'train': dataset.df[(validation_size + test_size + blackbox_size):]})

        for category, df in dfs.items():
            split_datasets.add(
                Dataset(klass=dataset.klass, branches=dataset.branches, category=category, df=df))

        return split_datasets

    @classmethod
    def merge_dataframes(cls, dataset_list):
        dataframes = [dataset.df for dataset in dataset_list]
        merged_df = reduce(lambda left, right: pd.merge(left, right, how='outer'), dataframes)
        return merged_df

    @classmethod
    def merge_by_category(cls, set_of_datasets):
        datasets_by_category = {}
        for dataset in set_of_datasets:
            if dataset.category not in datasets_by_category.keys():
                datasets_by_category.update({dataset.category: []})
            datasets_by_category[dataset.category].append(dataset)

        merged_datasets = set()
        for category, datasets in datasets_by_category.items():
            branches = datasets[0].branches
            merged_df = cls.merge_dataframes(datasets)
            merged_datasets.add(cls(branches=branches, category=category, df=merged_df))

        return merged_datasets

    def __init__(self, klass=None, branches=None, category=None, win=None, winseed=None,
                 bed_file=None, fasta_file=None, text_input=None, df=None):
        self.branches = branches  # list of seq, cons or fold branches
        self.klass = klass  # e.g. positive or negative
        self.category = category  # predict, or train, validation, test or blackbox for separated datasets
        self.df = df if df is not None else pd.DataFrame()

        if bed_file:
            input_type = 'bed'
            self.df = self.read_in_bed(bed_file)
        elif fasta_file:
            input_type = 'fasta'
            self.df = self.read_in_fasta(fasta_file)
        elif text_input:
            input_type = 'fasta'
            self.df = self.read_in_text(text_input)

        if win:
            self.df = Dataset.apply_window(self.df, win, winseed, input_type)

    def read_in_bed(self, bed_file):
        df = pd.read_csv(bed_file, sep='\t', header=None)

        if len(df.columns) == 3:
            df.columns = ['chrom_name', 'seq_start', 'seq_end']
            df['strand_sign'] = ''
        elif len(df.columns) >= 6:
            df = df[[0, 1, 2, 5]]
            df.columns = ['chrom_name', 'seq_start', 'seq_end', 'strand_sign']
        else:
            raise UserInputError('Invalid format of a .bed file.')
        if self.klass:
            df['klass'] = self.klass

        # TODO is it possible to do that in one line using just boolean masking? (could not manage that...)
        def check_valid(row):
            return row if seq.is_valid_chr(row['chrom_name']) else None

        df = df.apply(check_valid, axis=1).dropna()
        return df

    def reduce(self, ratio, seed):
        np.random.seed(seed)
        np.random.shuffle(self.df.values)
        if ratio <= 1:
            # handle as a ratio
            last = int(self.df.shape[0] * ratio)
        elif ratio < len(self.df.values):
            # handle as a final size
            last = int(ratio)
        elif ratio >= len(self.df.values):
            # keep the full dataset
            last = len(self.df.values)-1
        self.df = self.df[:last]
        return self

    def labels(self, alphabet=None):
        labels = self.df['klass']
        if alphabet:
            encoded_labels = [seq.translate(item, alphabet) for item in labels]
            return np.array(encoded_labels)
        else:
            return np.array(labels)

    def map_to_branches(self, references, alphabet, strand, outfile_path, ncpu=None):
        dna = alphabet == 'DNA'
        for branch in self.branches:
            if branch == 'seq':
                if 'input_seq' in self.df.columns:
                    self.df['seq'] = self.df['input_seq']
                    self.encode_col('seq', alphabet)
                else:
                    logger.info(f'Mapping intervals to the fasta reference...')
                    self.df = Dataset.map_to_fasta_dict(self.df, branch, references[branch], alphabet, strand)
            elif branch == 'cons':
                logger.info(f'Mapping sequences to the wig reference...')
                self.df = Dataset.map_to_wig(branch, self.df, references[branch])
            elif branch == 'fold':
                logger.info(f'Mapping and folding the sequences...')
                if 'input_seq' in self.df.columns:
                    self.df['fold'] = self.df['input_seq']
                    key_cols = ['input_seq']
                else:
                    self.df = Dataset.map_to_fasta_dict(self.df, branch, references[branch], alphabet, strand)
                    key_cols = ['chrom_name', 'seq_start', 'seq_end', 'strand_sign']

                self.fold_branch(key_cols, ncpu, dna=dna)

        self.save_to_file(outfile_path, do_zip=True)
        return self

    def sort_datapoints(self):
        self.df = self.df.sort_values(by=['chrom_name', 'seq_start'])
        return self

    def save_to_file(self, outfile_path, do_zip=False, ignore_cols=None):
        to_export = [col for col in self.df.columns if col not in ignore_cols] if ignore_cols else self.df.columns
        self.df.to_csv(outfile_path, sep='\t', columns=to_export, index=False)

        if do_zip:
            logger.info(f'Compressing dataset file...')
            zipped = ZipFile(f'{outfile_path}.zip', 'w')
            zipped.write(outfile_path, os.path.basename(outfile_path), compress_type=ZIP_DEFLATED)
            zipped.close()
            os.remove(outfile_path)

    def encode_col(self, col, alphabet):
        def map(row):
            sequence = row[col]
            encoded = Dataset.encode_sequence(sequence, alphabet)
            row[col] = self.sequence_to_string(encoded)
            return row

        self.df = self.df.apply(map, axis=1)
        return self

    @staticmethod
    def read_in_fasta(fasta_file):
        df = pd.DataFrame()

        with open(fasta_file, 'r') as file:
            header = None
            sequence = ""
            for line in file:
                if '>' in line:
                    # Save finished previous key value pair (unless it's the first iteration)
                    if header:
                        new_row = pd.DataFrame([[header, sequence]])
                        df = df.append(new_row)
                    header = line.strip().strip('>')
                    sequence = ""
                else:
                    if header:
                        sequence += line.strip()
                    else:
                        raise UserInputError("Provided reference file does not start with '>' fasta identifier.")

        df.columns = ['header', 'input_seq']
        return df

    @staticmethod
    def read_in_text(text):
        df = pd.DataFrame()
        seq_series = pd.Series(text.strip().split('\n'))
        df['input_seq'] = seq_series
        return df

    @staticmethod
    def map_to_fasta_dict(df, branch, ref_dictionary, alphabet, strand):
        # Returns only successfully mapped samples
        old_shape = df.shape[0]
        df[branch] = None

        def map(row):
            if row['chrom_name'] in ref_dictionary.keys():
                sequence = []
                for i in range(row['seq_start'], row['seq_end']):
                    sequence.append(ref_dictionary[row['chrom_name']][i])
                if strand and row['strand_sign'] == '-':
                    sequence = seq.complement(sequence, seq.COMPLEMENTARY[alphabet])
                if branch == 'seq':
                    row[branch] = Dataset.sequence_to_string(Dataset.encode_sequence(sequence, alphabet))
                elif branch == 'fold':
                    #  only temporary value for folding (won't be saved like this to file)
                    row[branch] = ''.join(sequence)
            else:
                row[branch] = None
            return row

        df = df.apply(map, axis=1)
        df.dropna(subset=[branch], inplace=True)

        portion = round((df.shape[0] / old_shape * 100), 2)
        logger.info(f'Successfully mapped {portion}% of samples.')
        return df

    @staticmethod
    def encode_sequence(sequence, alphabet):
        encoding = seq.onehot_encode_alphabet(seq.ALPHABETS[alphabet])
        new_sequence = [seq.translate(item, encoding) for item in sequence]
        return new_sequence

    @staticmethod
    def sequence_to_string(seq_list):
        #  temporary solution, as pandas can not save lists nor ndarrays into values
        string = ""
        for e in seq_list:
            if (type(e) == np.ndarray) or (type(e) == list):
                substring = ""
                for ee in e:
                    substring += str(ee) + ','
                substring = substring.strip(',')
                substring += '|'
                string += substring
            else:
                string += str(e) + '|'
        return string.strip('|').strip(',')

    @staticmethod
    def sequence_from_string(string):
        # TODO ideally make more explicit, maybe split the method
        #  (currently expects floats everywhere, can ever be soemthing else?)
        parts = string.strip().split('|')
        sequence = []

        for part in parts:
            if ',' in part:  # expects one hot encoded sequence, thus branches seq and fold
                subparts = part.strip().split(',')
                new_part = []
                for subpart in subparts:
                    new_part.append(float(subpart))
                sequence.append(np.array(new_part))
            else:  # expects not encoded sequence, thus branch cons
                sequence.append(np.array([float(part)]))

        return np.array(sequence)

    @staticmethod
    def map_to_wig(branch, df, ref_folder):
        not_found_chrs = set()
        chrom_files = f.list_files_in_dir(ref_folder, 'wig')

        current_file = None
        zipped = None
        current_chr = None
        current_header = {}
        parsed_line = {}
        df_copy = df.copy()

        for i, row in df_copy.iterrows():
            score = []

            if row['chrom_name'] and row['chrom_name'] == current_chr:
                result = Dataset.map_datapoint_to_wig(
                    score, zipped, row['seq_start'], row['seq_end'], current_file, current_header, parsed_line)
                if result:
                    score, current_header, parsed_line = result
                else:
                    # Covering the case when we reach end of reference file while still having some samples with current_chr not mapped
                    not_found_chrs.add(current_chr)
                    continue
            elif row['chrom_name'] in not_found_chrs:
                continue
            else:
                # When reading from new reference file
                files = list(filter(lambda f: f"{row['chrom_name']}." in os.path.basename(f), chrom_files))
                if len(files) == 1:
                    if current_file:
                        current_file.close()
                    current_chr = row['chrom_name']

                    current_file = f.unzip_if_zipped(files[0])
                    if '.gz' in files[0] or '.zip' in files[0]:
                        zipped = True
                    else:
                        zipped = False

                    line = f.read_decoded_line(current_file, zipped)
                    # Expecting first line of the file to be a header
                    if 'chrom' in line:
                        current_header = seq.parse_wig_header(line)
                    else:
                        raise UserInputError('File not starting with a proper wig header.')
                    result = Dataset.map_datapoint_to_wig(
                        score, zipped, row['seq_start'], row['seq_end'], current_file, current_header,
                        parsed_line)
                    if result:
                        score, current_header, parsed_line = result
                    else:  # should not happen here, as it's beginning of the file
                        not_found_chrs.add(current_chr)
                        continue
                else:
                    not_found_chrs.add(row['chrom_name'])
                    if len(files) == 0:
                        # TODO or rather raise an exception to let user fix it?
                        # Anyway, let the user know if none were found, thus the path given is wrong (currently it looks like it went through)
                        logger.warning(
                            f"Didn\'t find appropriate conservation file for {row['chrom_name']}, skipping the chromosome.")
                    else:  # len(files) > 1
                        logger.warning(f"Found multiple conservation files for {row['chrom_name']}, skipping the chromosome.")
                    continue

            if score and len(score) == (row['seq_end'] - row['seq_start']):
                # Score may be fully or partially missing if the coordinates are not part of the reference
                df.loc[i, branch] = Dataset.sequence_to_string(score)

        df.dropna(subset=[branch], inplace=True)
        return df

    @staticmethod
    def map_datapoint_to_wig(score, zipped, dp_start, dp_end, current_file, current_header, parsed_line):
        # TODO check the logic of passing on the parsed line
        # FIXME not covered: when the end of ref file is reached while there are still some samples from that file
        # unmapped - it reads empty lines - break with first empty line read as expecting it to be EOF? and move to
        # next chromosome somehow

        line = f.read_decoded_line(current_file, zipped)
        new_score = []
        if not line: return None

        if 'chrom' in line:
            current_header = seq.parse_wig_header(line)
            result = Dataset.map_datapoint_to_wig(score, zipped, dp_start, dp_end, current_file, current_header, parsed_line)
            if result:
                new_score, current_header, parsed_line = result
            else:
                return None
        else:
            if dp_start < current_header['start']:
                # Missed beginning of the datapoint while reading through the reference (should not happen)
                pass
            else:
                current_header, parsed_line = seq.parse_wig_line(line, current_header)
                while dp_start not in parsed_line.keys():
                    line = f.read_decoded_line(current_file, zipped)
                    if not line: return None
                    if 'chrom' in line:
                        current_header = seq.parse_wig_header(line)
                    else:
                        current_header, parsed_line = seq.parse_wig_line(line, current_header)
                    if current_header['start'] > dp_end:
                        # Did not find datapoint coordinates in reference file, might happen
                        break
                else:
                    for i in range(0, (dp_end - dp_start)):
                        coord = dp_start + i
                        if coord in parsed_line.keys():
                            score.append(parsed_line[coord])
                        else:
                            line = f.read_decoded_line(current_file, zipped)
                            if not line: return None
                            if 'chrom' in line:
                                current_header = seq.parse_wig_header(line)
                                line = f.read_decoded_line(current_file, zipped)
                                if not line: return None
                            current_header, parsed_line = seq.parse_wig_line(line, current_header)
                            try:
                                score.append(parsed_line[coord])
                            except:
                                # Lost the score in the reference in the middle of a datapoint, might happen
                                break

        return [score + new_score, current_header, parsed_line]

    def fold_branch(self, key_cols, ncpu=1, dna=True):
        # TODO check output, it's suspiciously quick for large numbers of samples
        tmp_dir = tempfile.gettempdir()
        original_length = self.df.shape[0]
        fasta_file = Dataset.dataframe_to_fasta(self.df, 'fold', key_cols, tmp_dir, f'df_{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')

        out_path = os.path.join(tmp_dir, f'folded_df_{str(datetime.datetime.now().strftime("%Y%m%d-%H%M"))}')
        out_file = open(out_path, 'w+')
        if dna:
            subprocess.run(['RNAfold', '--verbose', '--noPS', f'--jobs={ncpu}', fasta_file], stdout=out_file, check=True)
        else:
            subprocess.run(['RNAfold', '--verbose', '--noPS', '--noconv', f'--jobs={ncpu}', fasta_file], stdout=out_file,
                           check=True)

        out_file = open(out_path)
        lines = out_file.readlines()
        out_file.close()

        if (len(lines) / 3) == original_length:
            # The order should remain the same as long as --unordered is not set to True
            cols = list(self.df.columns)
            new_df = pd.DataFrame(columns=cols)
            fold_encoding = seq.onehot_encode_alphabet(['.', '|', 'x', '<', '>', '(', ')'])

            for i, line in enumerate(lines):
                # We're interested only in each third line in the output file (there are 3 lines per one input sequence)
                if (i + 1) % 3 == 0:
                    index = int(i/3)
                    new_df = new_df.append(self.df.iloc[index], ignore_index=True)
                    value = []
                    # line format: '.... (0.00)'
                    part1 = line.split(' ')[0].strip()
                    for char in part1:
                        value.append(seq.translate(char, fold_encoding))
                    new_df.loc[index, 'fold'] = Dataset.sequence_to_string(value)
            self.df = new_df
        else:
            logger.warning(f'Did not fold all the datapoints! (Only {len(lines)/3} out of {original_length}).')
            # We probably have no way to determine which were not folded if this happens
            raise ProcessError(f'Sorry, there was an error while trying to fold the given sequences.')

        return self

    @staticmethod
    def dataframe_to_fasta(df, branch, key_cols, path, name):
        path_to_fasta = os.path.join(path, (name + ".fa"))
        content = []

        def to_fasta(row):
            key = ""
            for col in key_cols:
                key += f'{row[col]}_'
            line1 = ">" + key.strip('_') + "\n"
            line2 = row[branch] + "\n"
            content.append(line1)
            content.append(line2)
            return row

        df.apply(to_fasta, axis=1)
        f.write(path_to_fasta, ''.join(content).strip())
        return path_to_fasta

    @staticmethod
    def apply_window(df, window_size, window_seed=64, type='bed'):
        np.random.seed(window_seed)

        def bed_window(row):
            length = row['seq_end'] - row['seq_start']
            if length > window_size:
                above = length - window_size
                rand = np.random.randint(0, above)
                new_start = row['seq_start'] + rand
                new_end = row['seq_start'] + rand + window_size
            elif length < window_size:
                missing = window_size - length
                rand = np.random.randint(0, missing)
                new_start = row['seq_start'] - rand
                new_end = row['seq_end'] + (missing - rand)
            else:
                new_start = row['seq_start']
                new_end = row['seq_end']

            row['seq_start'] = int(new_start)
            row['seq_end'] = int(new_end)
            return row

        def fasta_window(row):
            sequence = row['input_seq']
            length = len(sequence)
            if length > window_size:
                above = length - window_size
                rand = np.random.randint(0, above)
                new_sequence = sequence[rand:(rand+window_size)]
            elif length < window_size:
                missing = window_size - length
                rand = np.random.randint(0, missing)
                new_sequence = ('N' * rand) + sequence + ('N' * (missing-rand))
            else:
                new_sequence = sequence

            row['input_seq'] = new_sequence
            return row

        if type == 'bed':
            df = df.apply(bed_window, axis=1)
        elif type == 'fasta':
            df = df.apply(fasta_window, axis=1)

        return df
