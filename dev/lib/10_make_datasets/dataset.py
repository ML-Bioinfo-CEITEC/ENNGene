from ..utils import file_utils as f
from ..utils import sequence as seq


class Dataset:

    def __init__(self, branch, klass=None, bed_file=None, ref_dict=None, strand=None, encoding=None, datasetlist=None):
        self.branch = branch

        if datasetlist:
            self.dictionary = self.merge(datasetlist)
        else:
            self.dictionary = self.bed_to_dictionary(bed_file, ref_dict, strand, klass)

            if self.branch == 'fold':
                self.dictionary = self.fold_sequence(self.dictionary)

            if encoding:
                for key, arr in self.dictionary.items():
                    new_arr = [seq.translate(item, encoding) for item in arr]
                    self.dictionary.update({key: new_arr})

    def separate_by_chr(self, chr_list):
        separated_dataset = {}
        for key, sequence_list in self.dictionary.items():
            chromosome = key.split('_')[0]
            if chromosome in chr_list:
                separated_dataset.update({key: sequence_list})

        return separated_dataset

    # def add_value(self, value):
    #     new_dict = {}
    #     for key, sequence in self.dictionary.items():
    #         new_key = key + '_' + value
    #         new_dict.update({new_key: sequence})
    #     self.dictionary = new_dict
    #
    #     return self.dictionary

    def export_to_bed(self, path):
        return f.dictionary_to_bed(self.dictionary, path)

    @staticmethod
    def bed_to_dictionary(bed_file, ref_dictionary, strand, klass):
        file = f.filehandle_for(bed_file)
        seq_dict = {}

        for line in file:
            values = line.split()

            # 0 - chr. name, 1 - seq. start, 2 - seq. end, 5 - strand
            key = values[0] + "_" + values[1] + "_" + values[2] + "_" + values[5] + '_' + klass

            if values[0] in ref_dictionary.keys():
                start_position = int(values[1])
                end_position = (int(values[2])-1)
                sequence = ref_dictionary[values[0]][start_position:end_position]
                if strand and values[5] == '-':
                    sequence = seq.complement(sequence, seq.DNA_COMPLEMENTARY)

            if key and sequence:
                seq_dict.update({key: sequence.split()})

        return seq_dict

    @staticmethod
    def merge(list_of_datasets):
        return merged_dataset

    @staticmethod
    def fold_sequence(result_dict):
        return result_dict
