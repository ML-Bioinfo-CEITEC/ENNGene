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
    def split_random(cls, dataset, ratio_list, seed=56):
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
    def merge_by_category(cls, set_of_datasets):
        datasets_by_category = {}
        for dataset in set_of_datasets:
            if dataset.category not in datasets_by_category.keys(): datasets_by_category.update({dataset.category: []})
            datasets_by_category[dataset.category].append(dataset)

        final_datasets = set()
        for category, datasets in datasets_by_category.items():
            merged_datapoint_list = []
            for dataset in datasets:
                merged_datapoint_list += dataset.datapoint_list
            final_datasets.add(
                cls(branches=datasets[0].branches, category=category, datapoint_list=merged_datapoint_list))

        return final_datasets

    def __init__(self, klass=None, branches=None, category=None, bed_file=None, ref_files=None, strand=None, encoding=None,
                 datapoint_list=None):
        self.branches = branches  # list of seq, cons or fold branches
        self.klass = klass  # e.g. positive or negative
        self.category = category  # train, validation, test or blackbox for separated datasets

        # TODO complementarity currently applied only to sequence. Does the conservation score depend on strand?
        if datapoint_list:
            self.datapoint_list = datapoint_list
        else:
            logger.debug('Mapping bed file for klass {} onto {} reference files.'.format(self.klass, len(branches)))
            self.datapoint_list = self.map_bed_to_refs(branches, klass, bed_file, ref_files, encoding, strand)
            logger.debug('Finished mapping the file.')

        if 'fold' in self.branches and not datapoint_list:
            logger.debug('Folding sequences...')
            # can the result really be a dictionary? probably should
            file_name = 'fold' + '_' + klass
            # TODO probably the input may not be DNA, should the user define it? Or should we check it somewhere?
            self.datapoint_list = self.fold_branch(self.datapoint_list, file_name, dna=True)
            logger.debug('Finished folding sequences.')

    def save_to_file(self, dir_path, file_name):
        content = ""
        content += 'key' + "\t" + "\t".join(self.branches) + "\n"
        for datapoint in self.datapoint_list:
            content += datapoint.key() + "\t"
            for branch in self.branches:
                content += datapoint.string_value(branch) + "\t"
            content += "\n"

        file_path = os.path.join(dir_path, file_name)
        f.write(file_path, content.strip())
        return file_path

    def reduce(self, ratio, seed=84):
        random.seed(seed)
        random.shuffle(self.datapoint_list)
        last = int(len(self.datapoint_list) * ratio)

        reduced_dp_list = self.datapoint_list[0:last]

        self.datapoint_list = reduced_dp_list
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

    def fold_branch(self, datapoint_list, name, dna=True):
        tmp_dir = tempfile.gettempdir()
        fasta_file = self.datapoints_to_fasta(datapoint_list, 'fold', tmp_dir, name)

        out_path = os.path.join(tmp_dir, name + "_folded")
        out_file = open(out_path, 'w+')
        if dna:
            subprocess.run(["RNAfold", "--noPS", "--jobs=10", fasta_file], stdout=out_file, check=True)
        else:
            subprocess.run(["RNAfold", "--noPS", "--noconv", "--jobs=10", fasta_file], stdout=out_file, check=True)

        # TODO zip output file? can it help when reading large file??
        out_file = open(out_path)
        lines = out_file.readlines()
        out_file.close()

        if (len(lines) / 3) == len(datapoint_list):
            updated_datapoint_list = []
            for i, line in enumerate(lines):
                # TODO check that the order of datapoints is ok
                # We're interested only in each third line in the output file (there are 3 lines per one input sequence)
                if (i+1) % 3 == 0:
                    datapoint = datapoint_list[int(i/3)]
                    value = []
                    # line format: '.... (  0.00)'
                    part1 = line.split('(')[0].strip()
                    for char in part1:
                        value.append(char)
                    datapoint.branches_values.update({'fold': np.array(value)})
                    updated_datapoint_list.append(datapoint)
        else:
            raise Exception('Did not fold all the datapoints!')
            sys.exit()

        return updated_datapoint_list

    # def export_to_bed(self, path):
    #     return f.dictionary_to_bed(self.dictionary, path)
    #
    # def export_to_fasta(self, path):
    #     return f.dictionary_to_fasta(self.dictionary, path)

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

    @staticmethod
    def map_bed_to_refs(branches, klass, bed_file, ref_files, encoding, strand):
        file = f.filehandle_for(bed_file)
        datapoint_list = []

        for line in file:
            values = line.split()

            chrom_name = values[0]
            seq_start = values[1]
            seq_end = values[2]
            try:
                strand_sign = values[5]
            except:
                strand_sign = None
            branches_values = {}

            for branch in branches:
                # TODO adjust so that it gets translated only once for seq and branch together
                complement = branch == 'seq' or branch == 'fold'
                ref_dictionary = ref_files[branch]

                if chrom_name in ref_dictionary.keys():
                    # first position in chr in bed file is assigned as 0 (thus it fits the python indexing from 0)
                    start_position = int(seq_start)
                    # both bed file coordinates and python range exclude the last position
                    end_position = int(seq_end)
                    sequence = []
                    for i in range(start_position, end_position):
                        sequence.append(ref_dictionary[chrom_name][i])

                    if complement and strand and strand_sign == '-':
                        sequence = seq.complement(sequence, seq.DNA_COMPLEMENTARY)

                    if encoding and branch == 'seq':
                        sequence = [seq.translate(item, encoding) for item in sequence]

                    branches_values.update({branch: np.array(sequence)})

            # Save the point only if the value is available for all the branches
            if len(branches_values) == len(branches):
                datapoint_list.append(
                    DataPoint(branches, klass, chrom_name, seq_start, seq_end, strand_sign, branches_values))

        return datapoint_list
