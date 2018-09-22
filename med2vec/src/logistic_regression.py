'''
python2 ./src/logistic_regression.py --train_file ./data/training_data_1.txt   --test_file ./data/test_data_1.txt
'''

import sys
import argparse
import os
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help=" train file ", type=str)
parser.add_argument("--test_file", help="test file", type=str)
args = parser.parse_args()

dim = 1870  ############
def list2nparray(seqs, dim):
	arr = np.zeros((1,dim), dtype = float)
	for i in seqs:
		arr[0,i] += 1
	return arr 
t1 = time()
train_file = open(args.train_file, 'r')
train_lines = train_file.readlines()[1:]
f1 = lambda x:1 if 'True' in x else 0 
train_label = []
train_data = np.zeros((len(train_lines), dim), dtype=float)
for k,line in enumerate(train_lines):
	tokens = line.rstrip().split('\t')
	train_label.append(f1(tokens[0]))
	seqs = tokens[2].split()
	seqs = [int(i) for i in seqs]
	train_data[k,:] = list2nparray(seqs, dim)
train_label = np.array(train_label)

test_file = open(args.test_file, 'r')
test_lines = test_file.readlines()[1:]
f1 = lambda x:1 if 'True' in x else 0 
test_label = []
test_data = np.zeros((len(test_lines), dim), dtype=float)
for k,line in enumerate(test_lines):
	tokens = line.rstrip().split('\t')
	test_label.append(f1(tokens[0]))
	seqs = tokens[2].split()
	seqs = [int(i) for i in seqs]
	test_data[k,:] = list2nparray(seqs, dim)


################## Modelling
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(train_data, train_label)
prediction = lr.predict_proba(test_data)
#print(a[:10])
prediction = list(prediction[:,1])
print(roc_auc_score(test_label, prediction))
print(time()-t1)









