import numpy as np
import os
import platform
import random
import subprocess
import tempfile

from .data_point import DataPoint
from . import file_utils as f
from . import sequence as seq

import logging

logger = logging.getLogger('main')


class Dataset:

    @classmethod
    def load_from_file(cls, file_path):
        file = open(file_path)

        head = file.readline().strip()
        branches = head.split("\t")[1:]
        category = os.path.basename(file_path)

        datapoint_list = []
        for line in file:
            key, *values = line.strip().split("\t")
            branches_string_values = {}
            for i, value in enumerate(values):
                branches_string_values.update({branches[i]: value})
            datapoint_list.append(DataPoint.load(key, branches_string_values))

        return cls(branches=branches, category=category, datapoint_list=datapoint_list)

    @classmethod
    def split_by_chr(cls, dataset, chrs_by_category):
        categories_by_chr = cls.reverse_chrs_dictionary(chrs_by_category)

        # separate original dictionary by categories
        split_sets = {}
        for datapoint in dataset.datapoint_list:
            category = categories_by_chr[datapoint.chrom_name]
            if category not in split_sets.keys(): split_sets.update({category: []})
            split_sets[category].append(datapoint)

        # create Dataset objects from separated dictionaries
        final_datasets = set()
        for category, dp_list in split_sets.items():
            final_datasets.add(
                Dataset(klass=dataset.klass, branches=dataset.branches, category=category, datapoint_list=dp_list))

        return final_datasets

    @classmethod
    def reverse_chrs_dictionary(cls, dictionary):
        reversed_dict = {}
        for key, chrs in dictionary.items():
            for chr in chrs:
                reversed_dict.update({chr: key})

        return reversed_dict

    @classmethod
    def split_random(cls, dataset, ratio_list, seed):
        # so far the categories are fixed, not sure if there would be need for custom categories
        categories_ratio = {'train': float(ratio_list[0]),
                            'validation': float(ratio_list[1]),
                            'test': float(ratio_list[2]),
                            'blackbox': float(ratio_list[3])}

        random.seed(seed)
        random.shuffle(dataset.datapoint_list)
        dataset_size = len(dataset.datapoint_list)
        total = sum(categories_ratio.values())
        start = 0
        end = 0

        # TODO ? to assure whole numbers, we round down the division, which leads to lost of several samples. Fix it?
        split_datasets = set()
        for category, ratio in categories_ratio.items():
            size = int(dataset_size * ratio / total)
            end += (size - 1)
            dp_list = dataset.datapoint_list[start:end]

            split_datasets.add(
                Dataset(klass=dataset.klass, branches=dataset.branches, category=category, datapoint_list=dp_list))
            start += size

        return split_datasets

    @classmethod
    def merge_by_category(cls, set_of_datasets, outdir_path):
        datasets_by_category = {}
        for dataset in set_of_datasets:
            if dataset.category not in datasets_by_category.keys(): datasets_by_category.update({dataset.category: []})
            datasets_by_category[dataset.category].append(dataset)

        final_datasets = set()
        for category, datasets in datasets_by_category.items():
            file_path = os.path.join(outdir_path, category)
            branches = datasets[0].branches
            out_file = cls.initialize_file(file_path, branches)
            merged_datapoint_list = []
            for dataset in datasets:
                merged_datapoint_list += dataset.datapoint_list
            final_datasets.add(
                cls(branches=branches, category=category, datapoint_list=merged_datapoint_list))
            for datapoint in merged_datapoint_list:
                datapoint.write(out_file)
            out_file.close()

        return final_datasets

    def __init__(self, klass=None, branches=None, category=None, bed_file=None, win=None, winseed=None,
                 datapoint_list=[]):
        self.branches = branches  # list of seq, cons or fold branches
        self.klass = klass  # e.g. positive or negative
        self.category = category  # train, validation, test or blackbox for separated datasets
        self.datapoint_list = datapoint_list

        if bed_file and win:
            self.datapoint_list = self.read_in_bed(bed_file, win, winseed)

    def read_in_bed(self, bed_file, window, window_seed):
        datapoint_list = []
        file = open(bed_file)

        for line in file:
            values = line.split()

            chrom_name = values[0]
            # first position in chr in bed file is assigned as 0 (thus it fits the python indexing from 0)
            seq_start = int(values[1])
            # both bed file coordinates and python range exclude the last position
            seq_end = int(values[2])
            if len(values) >= 6:
                strand_sign = values[5]
            else:
                strand_sign = None

            datapoint = DataPoint(self.branches, self.klass, chrom_name, seq_start, seq_end, strand_sign,
                                  win=window, winseed=window_seed)
            datapoint_list.append(datapoint)

        return datapoint_list

    def reduce(self, ratio, seed):
        random.seed(seed)
        random.shuffle(self.datapoint_list)
        last = int(len(self.datapoint_list) * ratio)

        self.datapoint_list = self.datapoint_list[0:last]
        return self

    def values(self, branch):
        # return ordered list of values of datapoints
        values = []
        for datapoint in self.datapoint_list:
            values.append(datapoint.branches_values[branch])

        return np.array(values)

    def labels(self, alphabet=None):
        # return ordered list of values of datapoints
        labels = []
        for datapoint in self.datapoint_list:
            labels.append(datapoint.klass)

        if alphabet:
            encoded_labels = [seq.translate(item, alphabet) for item in labels]
            return np.array(encoded_labels)
        else:
            return np.array(labels)

    def map_to_branches(self, references, encoding, strand, outfile_path):
        # TODO directly zip the files
        out_file = Dataset.initialize_file(outfile_path, self.branches)

        for branch in self.branches:
            # TODO complementarity currently applied only to sequence. Does the conservation score depend on strand?
            reference = references[branch]
            if branch == 'seq':
                self.datapoint_list = self.map_to_fasta_dict(self.datapoint_list, branch, reference, encoding, strand)
            elif branch == 'cons':
                self.datapoint_list = self.map_to_wig(self.datapoint_list, reference)
            elif branch == 'fold':
                datapoint_list = self.map_to_fasta_dict(self.datapoint_list, branch, reference, False, strand)
                # FIXME does not save result of the proper length
                logger.debug('Folding sequences in {} dataset...'.format(self.category))
                file_name = 'fold' + '_' + self.category
                # TODO probably the input may not be DNA, should the user define it? Or should we check it somewhere?
                self.datapoint_list = self.fold_branch(file_name, datapoint_list, dna=True)

        for datapoint in self.datapoint_list:
            datapoint.write(out_file)
        out_file.close()

        return self

    @staticmethod
    def map_to_fasta_dict(datapoint_list, branch, ref_dictionary, encoding, strand):
        # Returns only successfully mapped datapoints
        updated_datapoint_list = []
        for datapoint in datapoint_list:
            if datapoint.chrom_name in ref_dictionary.keys():
                sequence = []
                for i in range(datapoint.seq_start, datapoint.seq_end):
                    sequence.append(ref_dictionary[datapoint.chrom_name][i])

                if strand and datapoint.strand_sign == '-':
                    sequence = seq.complement(sequence, seq.DNA_COMPLEMENTARY)

                if encoding:
                    sequence = [seq.translate(item, encoding) for item in sequence]

                datapoint.branches_values.update({branch: np.array(sequence)})
                updated_datapoint_list.append(datapoint)

        return updated_datapoint_list

    @staticmethod
    def map_to_wig(datapoint_list, ref_folder, complement):
        # wig ref is just a path to the folder
        # remove or use seq.wig_to_dict

        # Returns only successfully mapped datapoints
        updated_datapoint_list = []
        for datapoint in datapoint_list:
            pass

        return updated_datapoint_list

    @staticmethod
    def fold_branch(file_name, datapoint_list, dna=True):
        tmp_dir = tempfile.gettempdir()
        fasta_file = Dataset.datapoints_to_fasta(datapoint_list, 'fold', tmp_dir, file_name)

        out_path = os.path.join(tmp_dir, file_name + "_folded")
        out_file = open(out_path, 'w+')
        if dna:
            # TODO mustard converts DNA to RNA also on its own, ask why not to use the --noconv option instead
            # TODO use ncpu option to define how many cores to use
            subprocess.run(["RNAfold", "--noPS", "--jobs=10", fasta_file], stdout=out_file, check=True)
        else:
            subprocess.run(["RNAfold", "--noPS", "--noconv", "--jobs=10", fasta_file], stdout=out_file, check=True)

        out_file = open(out_path)
        lines = out_file.readlines()
        out_file.close()

        if (len(lines) / 3) == len(datapoint_list):
            # The order should remain the same as long as --unordered is not set to True
            updated_datapoint_list = []
            for i, line in enumerate(lines):
                # TODO if the key is part of the output file, we could read it and by that identify the right datapoint
                # and update it, this way we eould not have to worry about the order of results, but it could be too slow
                # TODO what information do we use for training? The sequence, the MFE or something else?

                # We're interested only in each third line in the output file (there are 3 lines per one input sequence)
                if (i + 1) % 3 == 0:
                    datapoint = datapoint_list[int(i / 3)]
                    value = []
                    # line format: '.... (  0.00)'
                    part1 = line.split('(')[0].strip()
                    for char in part1:
                        value.append(char)
                    datapoint.branches_values.update({'fold': np.array(value)})
                    updated_datapoint_list.append(datapoint)
        else:
            raise Exception('Did not fold all the datapoints!')
            # We have no way to determine which were not folded if this happens
            sys.exit()

        return updated_datapoint_list

    @staticmethod
    def initialize_file(path, branches):
        out_file = open(path, 'w')
        header = 'key' + '\t' + '\t'.join(branches) + '\n'
        out_file.write(header)
        return out_file

    @staticmethod
    def datapoints_to_fasta(datapoint_list, branch, path, name):
        path_to_fasta = os.path.join(path, (name + ".fa"))
        content = ""
        for datapoint in datapoint_list:
            line1 = ">" + datapoint.key() + "\n"
            # Assuming the branch contains valid sequence (e.g. ['A', 'T', 'C', 'G']
            line2 = ''.join(datapoint.branches_values[branch]) + "\n"
            content += line1
            content += line2

        f.write(path_to_fasta, content.strip())
        return path_to_fasta
