import numpy as np
import pandas as pd

from heatmap import heatmap

#df = pd.read_csv('../logs/depression_class/leave_one_out_predictions.txt', names=['Participant','Correct','Prediction','Votes','Total','Confidence'])
df = pd.read_csv('../logs/control_vs_condition/leave_one_out_predictions.txt', names=['Participant','Correct','Prediction','Votes','Total','Confidence'])

#labels = ['Normal', 'Mild', 'Moderate']
labels = ['Control', 'Condition']

matrix = []

for i in range(0, len(labels)):
    print(f'--------{labels[i]}--------')

    TP = len(df.query(f'Correct == {i} and Prediction == {i}'))
    FP = len(df.query(f'Correct != {i} and Prediction == {i}'))
    FN = len(df.query(f'Correct == {i} and Prediction != {i}'))
    TN = len(df.query(f'Correct != {i} and Prediction != {i}'))

    try:
        acc = (TP + TN) / (TP + TN + FN + FP)
        print(f'Accuracy: \t({TP} + {TN}) / ({TP} + {TN} + {FN} + {FP}) \t= {acc}')

        prec = TP / (TP + FP)
        print(f'Precision: \t{TP} / ({TP} + {FP}) \t\t\t= {prec}')

        rec = TP / (TP + FN)
        print(f'Recall: \t{TP} / ({TP} + {FN}) \t\t\t= {rec}')

        spec = TN / (TN + FP)
        print(f'Specificity: \t{TN} / ({TN} + {FP}) \t\t\t= {spec}')

        f1 = 2 * ((prec * rec) / (prec + rec))
        print(f'F1: \t\t2*((prec * rec) / (prec + rec)) = {f1}')
    except:
        pass

    t = []

    for j in range(0, len(labels)):
        t.append(len(df.query(f'Correct == {i} and Prediction == {j}')))

    
    matrix.append(t)

matrix = np.array(matrix)

heatmap(matrix, xticklabels=labels, yticklabels=labels, filename='leave_one_out.pdf')
