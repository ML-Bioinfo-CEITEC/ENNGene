#!/usr/bin/env bash
# Extract data from reference

# mkdir create separate folders
# cd

classes=('pos' 'neg')
branches=('seq' 'cons' 'fold')

for class in ${classes[@]}
    do
        make_dataset.py \
            --coord ${class}.bed \
            --ref genome.fa \
            --reftype fasta \
            --onehot [A,C,G,T,N] \
            > sequence_${class}_dataset.txt
        make_dataset.py \
            --coord ${class}.bed \
            --ref phastcons.bedgraph \
            --reftype bedgraph \
            --score \
            > conservation_${class}_dataset.txt
        fold_seq.py \
            --input sequence_${class}_dataset.txt \
            > fold_${class}_dataset.txt
    done


# Merge positives and negatives (classes) and add labels
for branch in ${branches[@]}
    do
        file_list=()
        for class in ${classes[@]}
            do
                table-paste-col \
                    --table ${branch}_${class}_dataset.txt \
                    --col-name "class" \
                    --col-val $class \
                > ${branch}_${class}_dataset.labelled.txt

                file_list+=${branch}_${class}_dataset.labelled.txt
            done

        files_to_merge=$(printf " %s" "${file_list[@]}")
        files_to_merge=${files_to_merge:1}
        table-cat $files_to_merge > ${branch}_dataset.txt

        rm files_to_merge
    done


# Separate test, train, validation etc
valid_chrs=('chr1' 'chr2' 'chr3' 'chr4' 'chr5' 'chr6' 'chr7' 'chr8' 'chr9' 'chr10' 'chr11' 'chr12' 'chr13' 'chr14' \
            'chr15' 'chr16' 'chr17' 'chr18' 'chr19' 'chr20' 'chr21' 'chr22' 'chrY' 'chrX' 'chrMT')
validation=('chr10')
test=('chr20')
blackbox=('chr21')
# array subtraction in bash?
# test=valid_chrs-validation-test-blackbox
train=('chr1' 'chr2' 'chr3' 'chr4' 'chr5' 'chr6' 'chr7' 'chr8' 'chr9' 'chr11' 'chr12' 'chr13' 'chr14' \
            'chr15' 'chr16' 'chr17' 'chr18' 'chr19' 'chr22' 'chrY' 'chrX' 'chrMT')

chromosomes=(['train']=train ['validation']=validation ['test']=test ['blackbox']=blackbox)
for branch in ${branches[@]}
    do
        for category in ${!chromosomes[@]}
            do
            separate_sets_by_chr.py \
                --input ${branch}_dataset \
                --chr ${chromosomes[$category]} \
            > ${branch}_${category}_dataset.txt
        done
    done

# ... to training

