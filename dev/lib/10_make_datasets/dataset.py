from ..utils import file_utils as f
from ..utils import sequence as seq


class Dataset:

    @classmethod
    def merge(cls, list_of_datasets):
        return merged_dataset

    def __init__(self, branch):
        self.branch = branch

    def create(self, bed_file, ref_dict, strand, encoding):
        result_dict = self.bed_to_dictionary(bed_file, ref_dict, strand)

        if self.branch == 'cons':
            self.fold_sequence(result_dict)

        if encoding:
            for key, arr in result_dict.items():
                new_arr = [seq.translate(item, encoding) for item in arr]
                result_dict.update({key: new_arr})

        return result_dict

    @staticmethod # TODO should be instance method on object dataset
    def separate_sets_by_chr(dataset, chr_list):
        # FIXME dataset should be the object itself, which means it should be created during initialization instead
        # of using separate create method
        separated_dataset = {}
        for key, sequence_list in dataset.items():
            chromosome = key.split('_')[0]
            if chromosome in chr_list:
                separated_dataset.update({key: sequence_list})

        return separated_dataset

    @staticmethod
    def bed_to_dictionary(bed_file, ref_dictionary, strand):
        file = f.filehandle_for(bed_file)
        seq_dict = {}

        for line in file:
            values = line.split()

            # 0 - chr. name, 1 - seq. start, 2 - seq. end, 5 - strand
            key = values[0] + "_" + values[1] + "_" + values[2] + "_" + values[5]

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
    def fold_sequence(result_dict):
        return
