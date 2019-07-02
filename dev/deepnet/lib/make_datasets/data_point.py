class DataPoint:

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
            if type(e) == list:
                substring = ""
                for el in e:
                    substring += str(el) + ", "
                substring = substring.strip(', ')
                substring += " | "
                string += substring
            else:
                string += str(e) + ", "

        return string.strip(' | ').strip(', ')
