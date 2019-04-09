


# Extract data from reference
foreach $class in ((pos neg))
do
    make_dataset.py \
        --coord $class.bed \
        --ref genome.fa \
        --reftype fasta \
        --onehot [A,C,G,T,N] \
        > sequence_$class_dataset.txt
    make_dataset.py \
        --coord $class.bed \
        --ref phastcons.bedgraph \
        --reftype bedgraph \
        --score \
        > conservation_$class_dataset.txt
    fold_seq.py \
        --input sequence_${class}_dataset.txt \
        > fold_$class_dataset.txt

done

# Merge positives and negatives (classes) and add labels
foreach $branch in ((seq, cons, fold))
do
    file_list = []
    foreach $class in ((pos neg))
    do
        table-paste-col \
            --table ${branch}_${class}_dataset.txt \
            --col-name class \
            --col-val $class \
        > ${branch}_${class}_dataset.labelled.txt
        
        file_list += "${branch}_${class}_dataset.labelled.txt"
    done
    
    $files_to_merge = join(" ", empty_file_list)
    
    table-cat $files_to_merge > ${branch}_dataset.txt

    # rm temp files in file_list

done

# Separate test, train, validation etc
chromosomes = {train = ["chr1", "chr2"], ... }
foreach $branch in branches
do
    foreach $category in keys chromosomes
    do
        separate_sets_by_chr.py \
            --input ${branch}_dataset \
            --chr @[$chromosomes{$category}] \
        > ${branch}_${category}_dataset.txt
    done
done

# ... to training

