import numpy as np
import pandas as pd

from heatmap import heatmap

df = pd.read_csv('../logs/control_vs_condition/leave_one_out_predictions.txt', names=['Participant','Correct','Prediction','Votes','Total','Confidence'])

TP = len(df.query('Correct == 1 and Prediction == 1'))
FP = len(df.query('Correct == 0 and Prediction == 1'))
TN = len(df.query('Correct == 0 and Prediction == 0'))
FN = len(df.query('Correct == 1 and Prediction == 0'))

matrix = np.array([[TP, FN], [FP, TN]])

heatmap(matrix, xticklabels=['Condition', 'Control'], yticklabels=['Condition', 'Control'], filename='leave_one_out.pdf')

acc = (TP + TN) / (TP + TN + FN + FP)
prec = TP / (TP + FP)
rec = TP / (TP + FN)
spec = TN / (TN + FP)
f1 = 2 * ((prec * rec) / (prec + rec))

print(f'Accuracy: {acc}')
print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'Specificity: {spec}')
print(f'F1: {f1}')