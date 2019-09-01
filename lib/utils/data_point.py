import numpy as np

from . import sequence as seq


class DataPoint:

    @classmethod
    def load(cls, key, string_value):
        value = cls.value_from_string(string_value)
        chrom_name, seq_start, seq_end, strand_sign, klass = cls.attrs_from_key(key)

        return cls(chrom_name, seq_start, seq_end, strand_sign, klass, value)

    @classmethod
    def value_from_string(cls, string_value):
        parts = string_value.split('\t')
        new_parts = []

        for part in parts:
            new_part = []
            subparts = part.split(',')
            for subpart in subparts:
                new_part.append(float(subpart))
            new_parts.append(np.array(new_part))

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

    def __init__(self, chrom_name, seq_start, seq_end, strand_sign, klass, value):
        self.chrom_name = chrom_name
        self.seq_start = seq_start
        self.seq_end = seq_end
        self.strand_sign = strand_sign  # may be None
        self.klass = klass
        self.value = value

    def key(self):
        try:
            key = self.chrom_name + "_" + self.seq_start + "_" + self.seq_end + "_" + self.strand_sign + '_' + self.klass
        except:
            key = self.chrom_name + "_" + self.seq_start + "_" + self.seq_end + '_' + self.klass
        return key

    def string_value(self):
        string = ""
        for e in self.value:
            if type(e) == np.ndarray:
                substring = ""
                for el in e:
                    substring += str(el) + ", "
                substring = substring.strip(', ')
                substring += "\t"
                string += substring
            else:
                string += str(e) + "\t"

        return string.strip('\t').strip(', ')

    def translate_value(self, encoding):
        new_value = np.array([seq.translate(item, encoding) for item in self.value])
        self.value = new_value
