#!/usr/bin/env bash

python deepnet.py make_datasets \
--coord "/home/eliska/workspace/deepnet/files/test_set/pos.bed" \
"/home/eliska/workspace/deepnet/files/test_set/neg.bed" \
--ref "/home/eliska/workspace/deepnet/files/test_set/Homo.fa" \
--branches 'seq' 'cons' \
--consdir "/home/eliska/workspace/deepnet/files/test_set/phyloP100way" \
--onehot 'C' 'G' 'T' 'A' 'N' \
--strand True