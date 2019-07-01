import random

from ..utils import file_utils as f
from ..utils import sequence as seq


class Dataset:

    @classmethod
    def separate_by_chr(cls, dataset, chrs_by_category):
        separated_dictionaries = {}
        datasets_dict = {}
        categories_by_chr = cls.reverse_chrs_dictionary(chrs_by_category)

        # separate original dictionary by categories
        for key, sequence_list in dataset.dictionary.items():
            chromosome = key.split('_')[0]
            try:
                category = categories_by_chr[chromosome]
                if category not in separated_dictionaries.keys(): separated_dictionaries.update({category: {}})
                separated_dictionaries[category].update({key: sequence_list})
            except:
                # probably unnecessary (already checked for valid chromosomes before?)
                continue

        # create Dataset objects from separated dictionaries
        for category, dict in separated_dictionaries.items():
            # TODO maybe unnecessary to use category as a key, as it's saved as datasets attribute
            datasets_dict.update({category: Dataset(dataset.branch, category=category, dictionary=dict)})

        return datasets_dict

    @classmethod
    def reverse_chrs_dictionary(cls, dictionary):
        reversed_dict = {}
        for key, chrs in dictionary.items():
            for chr in chrs:
                reversed_dict.update({chr: key})

        return reversed_dict

    @classmethod
    def separate_random(cls, dataset, ratio_list, seed):
        # so far the categories are fixed, not sure if there would be need for custom categories
        categories_ratio = {'test': float(ratio_list[0]),
                            'validation': float(ratio_list[1]),
                            'train': float(ratio_list[2]),
                            'blackbox': float(ratio_list[3])}

        random.seed(seed)
        randomized = list(dataset.dictionary.items())
        random.shuffle(randomized)

        dataset_size = len(dataset.dictionary)
        total = sum(categories_ratio.values())
        separated_datasets = {}
        start = 0; end = 0

        # TODO ? to assure whole numbers, we round down the division, which leads to lost of several samples. Fix it?
        for category, ratio in categories_ratio.items():
            size = int(dataset_size*ratio/total)
            end += (size-1)
            separated_datasets.update({category: dict(randomized[start:end])})
            start += size

        return separated_datasets

    @classmethod
    def merge(cls, list_of_datasets):
        merged_dictionary = {}
        branch = list_of_datasets[0].branch
        for dataset in list_of_datasets:
            merged_dictionary.update(dataset.dictionary)

        return cls(branch, dictionary=merged_dictionary)

    def __init__(self, branch, klass=None, category=None, bed_file=None, ref_dict=None, strand=None, encoding=None,
                 dictionary=None):
        self.branch = branch # seq, cons or fold
        self.klass = klass  # e.g. positive and negative
        self.category = category  # train, validation, test or blackbox

        # TODO is there a way a folding branch could use already converted datasets from seq branch, if available?
        # TODO complementarity currently applied only to sequence. Does the conservation score depend on strand?
        complement = branch == 'seq' or branch == 'fold'

        if dictionary:
            self.dictionary = dictionary
        else:
            self.dictionary = self.bed_to_dictionary(bed_file, ref_dict, strand, klass, complement)

        if self.branch == 'fold' and not dictionary:
            # can the result really be a dictionary? probably should
            file_name = branch + "_" + klass
            self.dictionary = seq.fold(self.dictionary, file_name)

        # TODO apply one-hot encoding also to the fold branch?
        if encoding and branch == 'seq':
            for key, arr in self.dictionary.items():
                new_arr = [seq.translate(item, encoding) for item in arr]
                self.dictionary.update({key: new_arr})

    # def export_to_bed(self, path):
    #     return f.dictionary_to_bed(self.dictionary, path)
    #
    # def export_to_fasta(self, path):
    #     return f.dictionary_to_fasta(self.dictionary, path)

    @staticmethod
    def bed_to_dictionary(bed_file, ref_dictionary, strand, klass, complement):
        file = f.filehandle_for(bed_file)
        final_dict = {}

        for line in file:
            values = line.split()

            chrom_name = values[0]
            seq_start = values[1]
            seq_end = values[2]
            strand_sign = None
            sequence = None

            # TODO implement as a standalone object with attributes chrom_name, seq_start, ...
            try:
                strand_sign = values[5]
                key = chrom_name + "_" + seq_start + "_" + seq_end + "_" + strand_sign + '_' + klass
            except:
                key = chrom_name + "_" + seq_start + "_" + seq_end + '_' + klass

            if chrom_name in ref_dictionary.keys():
                # first position in chromosome in bed file is assigned as 0 (thus it fits the python indexing from 0)
                start_position = int(seq_start)
                # both bed file coordinates and python range exclude the last position
                end_position = int(seq_end)
                sequence = []
                for i in range(start_position, end_position):
                    sequence.append(ref_dictionary[chrom_name][i])

                if complement and strand and strand_sign == '-':
                    sequence = seq.complement(sequence, seq.DNA_COMPLEMENTARY)

            if key and sequence:
                final_dict.update({key: sequence})

        return final_dict
