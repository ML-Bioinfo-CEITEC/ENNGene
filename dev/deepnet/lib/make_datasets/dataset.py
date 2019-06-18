from ..utils import file_utils as f
from ..utils import sequence as seq


class Dataset:

    def __init__(self, branch, klass=None, bed_file=None, ref_dict=None, strand=None, encoding=None, datasetlist=None):
        self.branch = branch

        if datasetlist:
            self.dictionary = self.merge(datasetlist)
        else:
            # TODO is there a way a folding branch could use already converted datasets from seq branch, if available?
            self.dictionary = self.bed_to_dictionary(bed_file, ref_dict, strand, klass)

            if self.branch == 'fold' and not datasetlist:
                # can the result really be a dictionary? probably should
                file_name = branch + "_" + klass
                self.dictionary = seq.fold(self.dictionary, file_name)

            if encoding:
                for key, arr in self.dictionary.items():
                    new_arr = [seq.translate(item, encoding) for item in arr]
                    self.dictionary.update({key: new_arr})

    # TODO allow random separation too
    def separate_by_chr(self, chr_list):
        separated_dataset = {}
        for key, sequence_list in self.dictionary.items():
            chromosome = key.split('_')[0]
            if chromosome in chr_list:
                separated_dataset.update({key: sequence_list})

        return separated_dataset

    # def export_to_bed(self, path):
    #     return f.dictionary_to_bed(self.dictionary, path)
    #
    # def export_to_fasta(self, path):
    #     return f.dictionary_to_fasta(self.dictionary, path)

    @staticmethod
    def bed_to_dictionary(bed_file, ref_dictionary, strand, klass):
        file = f.filehandle_for(bed_file)
        final_dict = {}

        for line in file:
            values = line.split()

            # 0 - chr. name, 1 - seq. start, 2 - seq. end, 5 - strand
            key = values[0] + "_" + values[1] + "_" + values[2] + "_" + values[5] + '_' + klass

            if values[0] in ref_dictionary.keys():
                # first position in chromosome in bed file is assigned as 0 (thus it fits the python indexing from 0)
                start_position = int(values[1])
                # both bed file coordinates and python when accessing the values in list exclude the last position
                end_position = int(values[2])
                sequence = ref_dictionary[values[0]][start_position:end_position]
                if strand and values[5] == '-':
                    sequence = seq.complement(sequence, seq.DNA_COMPLEMENTARY)

            if key and sequence:
                final_dict.update({key: sequence.split()})

        return final_dict

    @staticmethod
    def merge(list_of_datasets):
        merged_dictionary = {}
        for dataset in list_of_datasets:
            merged_dictionary.update(dataset.dictionary)

        return merged_dictionary


