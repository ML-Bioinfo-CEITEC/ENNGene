import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (accuracy_score, average_precision_score, 
    auc, classification_report, confusion_matrix, f1_score, 
    precision_recall_curve, precision_score, 
    recall_score, roc_auc_score, roc_curve)


# three functions that return evaluation plots for 2 and more classes
# evaluation confusion matrix, precision recall curve and ROC curve
# inputs that need to be added: 
#   labels_dict: dictionary of categorical labels and labels
#          e.g. {5: 'random', 3: 'LIN28B', 2: 'HNRNPA1', 4: 'TARDBP', 1: 'FXR1', 0: 'CPEB4'}
#   my_xy_ticks: list of labels created from labels_dict values
#          e.g. ['random', 'LIN28B', 'HNRNPA1', 'TARDBP', 'FXR1', 'CPEB4']


def plot_eval_cfm(y_true, y_pred, labels, output_dir_path):
    name = 'confusion_matrix'
    outfile_path = output_dir_path / name
    model_cfm = confusion_matrix(y_true, y_pred, labels=labels)

    figsize = (36, 30)

    cfm_sum = np.sum(model_cfm, axis=1, keepdims=True)
    cfm_perc = model_cfm / cfm_sum.astype(float) * 100
    annot = np.empty_like(model_cfm).astype(str)
    nrows, ncols = model_cfm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = model_cfm[i, j]
            p = cfm_perc[i, j]
            if i == j:
                s = cfm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
                
    cfm = pd.DataFrame(model_cfm)
    cfm.index.name = 'True'
    cfm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cfm, cmap= 'Reds', annot=annot, fmt='', ax=ax, annot_kws={"fontsize":14})
    ax.set_xlabel('Predicted', fontsize=20)
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(my_xy_ticks, rotation = 0, fontsize = 18)
    ax.xaxis.tick_top()
    ax.set_ylabel('True', fontsize=20)
    ax.set_yticklabels(my_xy_ticks, rotation = 360, fontsize = 18)
    
    plt.savefig(outfile_path, format='png')
    plt.show()
    
                                            
def plot_multiclass_prec_recall_curve(y_test, y_score, n_classes, output_dir_path):
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                         average="micro")

    name = 'multiclass_prec_recall_curve'
    outfile_path = output_dir_path / name
    
    plt.figure(figsize=(12, 12))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=3)
    lines.append(l)
    labels.append('micro-average Precision-recall (auc = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i in range(n_classes):
        l, = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append('Precision-recall for {0} (auc = {1:0.2f})'
                      ''.format(labels_dict[i], average_precision[i]))

    fig = plt.gcf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Multiclass Precision-Recall curve', fontsize=16)
    plt.legend(lines, labels, loc='lower left', prop=dict(size=14))
    plt.savefig(outfile_path, format='png')
    plt.show()
    
                      
def plot_multiclass_ROC_curve(y_test, y_score, n_classes, output_dir_path):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    name = 'multiclass_roc_curve'
    outfile_path = output_dir_path / name
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 12))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (auc = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=5)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=3,
                 label='ROC curve of {0} (auc = {1:0.2f})'
                 ''.format(labels_dict[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=3, alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1-specifity)', fontsize=14)
    plt.ylabel('True Positive Rate (sensitivity)', fontsize=14)
    plt.title('Multiclass Receiver operating characteristic', fontsize=16)
    plt.legend(loc="lower right", prop=dict(size=14))
    plt.savefig(outfile_path, format='png')
    plt.show()