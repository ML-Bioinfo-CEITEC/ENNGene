from ..utils import file_utils as f
from ..utils import sequence as seq


class Dataset:

    @classmethod
    def merge(cls, list_of_datasets):
        return merged_dataset

    def __init__(self, data_file, ref_dict, onehot=[]):
        # save other objects for multiple use
        self.data_file = data_file
        self.ref_dict = ref_dict
        self.one_hot = onehot

    def create(self, branch):
        self.data_file
        if branch == 'seq':
            do_something
        elif branch == 'cons':
            do_something
        elif branch == 'seq':
            do_something

        # if args.reftype == 'fasta':
        #     ref_dict = seq.fasta_to_dictionary(self.ref_file)
        #     result_dict = seq.bed_to_seq_dictionary(args.coord, ref_dict, args.strand)
        # else:
        #     # open reference file or folder
        #     # save in dictionary reference{chromosome}{position} = value (can be sequence or score)
        #     # open coordinates file
        #     coordFH = f.filehandle_for(args.coord)
        #     # for line in coordFH:
        #     # extract relevant positions from reference dictionary
        #     # values{name} = [values_per_position_in_range]
        #     # open coordinates file
        #     # make result_dict: coords{name} = "chr_start_stop_strand" (this is the key)
        #
        # if args.onehot:
        #     encoding = seq.encode_alphabet(args.onehot)
        #     for key, arr in result_dict.items():
        #         new_arr = [seq.translate(item, encoding) for item in arr]
        #         result_dict.update({key: new_arr})
        #
        # return result_dict
