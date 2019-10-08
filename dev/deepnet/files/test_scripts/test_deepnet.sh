python deepnet.py make_datasets --coord "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/files/test_set/pos.bed" \
                                        "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/files/test_set/neg.bed" \
                                --ref "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/files/test_set/Homo.fa" \
                                --branches 'seq' 'cons' \
                                --consdir "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/files/test_set/phyloP100way/" \
                                --onehot 'C' 'G' 'T' 'A' 'N' \
                                --strand True \
                                --split by_chr

python deepnet.py train --datasets "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/output/datasets/final_datasets/train" \
                                   "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/output/datasets/final_datasets/validation" \
                                   "/Users/danielkuchta/Desktop/Daniel/RBP/deepnet/output/datasets/final_datasets/test" \
                        --epochs 2 \
                        --hyper_tuning 'True' \
                        --branches 'seq' \
                        --experiment_name test \
                        --conv_num '2,3' \
                        --hyper_param_metric 'acc' \
                        --dense_num '2,4' \
                        --dense_units '2,5' \
                        --filter_num '[32,64]' \
                        --dropout '[0.2, 0.1]' \
                        --kernel_size '[2,4]' \
                        --tune_rounds 2