import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

def heatmap(matrix, title='Confusion Matrix', ylabel='True Label', xlabel='Predicted Label', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'], filename='conf_matrix.pdf'):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot=True,
                fmt="d")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename)


if __name__ == '__main__':
    true_positive = 3122
    false_positive = 13
    true_negative = 1677
    false_negative = 5
    
    matrix = np.array([[true_positive, false_negative], [false_positive, true_negative]])

    heatmap(matrix, xticklabels=['Control', 'Condition'], yticklabels=['Control', 'Condition'], filename='10ep.pdf')