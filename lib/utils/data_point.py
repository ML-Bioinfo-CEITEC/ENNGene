import numpy as np

from . import sequence as seq


class DataPoint:

    @classmethod
    def load(cls, key, branches_string_values):
        chrom_name, seq_start, seq_end, strand_sign, klass = cls.attrs_from_key(key)

        branches_values = {}
        for branch, string_value in branches_string_values.items():
            value = cls.value_from_string(string_value)
            branches_values.update({branch: value})

        return cls(branches_values.keys(), klass, chrom_name, seq_start, seq_end, strand_sign, branches_values)

    @classmethod
    def value_from_string(cls, string_value):
        parts = string_value.strip().split('|')
        new_parts = []

        for part in parts:
            if '[' in part:
                subparts = part.strip('[').strip(']').split()
                new_part = []
                for subpart in subparts:
                    new_part.append(float(subpart))
                new_parts.append(np.array(new_part))
            else:
                new_parts.append(float(part))

        return np.array(new_parts)

    @classmethod
    def attrs_from_key(cls, key):
        attrs = key.split('_')

        chrom_name = attrs[0]
        seq_start = attrs[1]
        seq_end = attrs[2]

        if len(attrs) == 4:
            strand_sign = None
            klass = attrs[3]
        else:
            strand_sign = attrs[3]
            klass = attrs[4]

        return [chrom_name, seq_start, seq_end, strand_sign, klass]

    def __init__(self, branches, klass, chrom_name, seq_start, seq_end, strand_sign, branches_values):
        self.branches = branches
        self.klass = klass
        self.chrom_name = chrom_name
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.strand_sign = strand_sign  # may be None
        self.branches_values = branches_values

    def key(self):
        try:
            key = self.chrom_name + "_" + self.seq_start + "_" + self.seq_end + "_" + self.strand_sign + '_' + self.klass
        except:
            key = self.chrom_name + "_" + self.seq_start + "_" + self.seq_end + '_' + self.klass
        return key

    def string_value(self, branch):
        string = ""
        for e in self.branches_values[branch]:
            if type(e) == list:
            # if type(e) == np.ndarray:
                substring = ""
                for el in e:
                    substring += str(el) + ", "
                substring = substring.strip(', ')
                substring += '|'
                string += substring
            else:
                string += str(e) + '|'

        return string.strip('|').strip(', ')
