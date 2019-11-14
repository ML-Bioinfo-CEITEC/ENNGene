import numpy as np
import random


class DataPoint:

    @classmethod
    def load(cls, key, branches_string_values):
        chrom_name, seq_start, seq_end, strand_sign, klass = cls.attrs_from_key(key)

        branches_values = {}
        for branch, string_value in branches_string_values.items():
            value = cls.value_from_string(string_value)
            branches_values.update({branch: value})

        return cls(branches_values.keys(), klass, chrom_name, int(seq_start), int(seq_end), strand_sign, branches_values=branches_values)

    @classmethod
    def value_from_string(cls, string_value):
        parts = string_value.strip().split('|')
        new_parts = []

        for part in parts:
            if ',' in part:
                subparts = part.strip().split(',')
                new_part = []
                for subpart in subparts:
                    new_part.append(float(subpart))
                new_parts.append(np.array(new_part))
            else:
                new_parts.append(np.array([float(part)]))

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

    def __init__(self, branches, klass, chrom_name, seq_start, seq_end, strand_sign, win=None, winseed=None,
                 branches_values=None):
        self.branches = branches
        self.klass = klass
        self.chrom_name = chrom_name
        self.strand_sign = strand_sign  # may be None
        self.branches_values = branches_values or {}

        if win:
            self.seq_start, self.seq_end = self.apply_window(seq_start, seq_end, win, winseed)
        else:
            self.seq_start = seq_start
            self.seq_end = seq_end

    def key(self):
        if self.strand_sign:
            key = self.chrom_name + "_" + str(self.seq_start) + "_" + str(self.seq_end) + "_" + self.strand_sign + '_' + self.klass
        else:
            key = self.chrom_name + "_" + str(self.seq_start) + "_" + str(self.seq_end) + '_' + self.klass
        return key

    def value(self, branch):
        return self.branches_values[branch]

    def string_value(self, branch):
        string = ""
        for e in self.value(branch):
            if type(e) == np.ndarray:
                substring = ""
                for el in e:
                    substring += str(el) + ", "
                substring = substring.strip(', ')
                substring += '|'
                string += substring
            else:
                string += str(e) + '|'

        return string.strip('|').strip(', ')

    def write(self, out_file, no_value=False):
        out_file.write(self.key() + '\t')

        # The novalue option is for interval files that do not yet contain any values
        if not no_value:
            for branch in self.branches:
                out_file.write(self.string_value(branch) + '\t')
        out_file.write('\n')

    @staticmethod
    def apply_window(seq_start, seq_end, window, seed=64):
        random.seed(seed)
        if (seq_end - seq_start) > window:
            above = (seq_end - seq_start) - window
            rand = random.randint(0, above)
            new_start = seq_start + rand
            new_end = seq_start + rand + window
        elif (seq_end - seq_start) < window:
            missing = window - (seq_end - seq_start)
            rand = random.randint(0, missing)
            new_start = seq_start - rand
            new_end = seq_end + (missing - rand)
        else:
            new_start = seq_start
            new_end = seq_end

        return new_start, new_end
