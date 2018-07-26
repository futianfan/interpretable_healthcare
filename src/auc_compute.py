import sys

from sklearn.metrics import roc_auc_score

with open(sys.argv[1], 'r') as fin:
    lines = fin.readlines()
    y_pred = []
    y_label = []
    for line in lines:
        y_pred.append(float(line.split()[1]))
        y_label.append(1 - int(line.split()[0]))

print(roc_auc_score(y_label, y_pred))
