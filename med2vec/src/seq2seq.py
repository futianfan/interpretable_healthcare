# -*- coding: utf-8 -*-
'''
python src/seq2seq.py --data_file ./data/train_data --word2vec ./data/training_model_by_word2vec_1.vector
'''

import argparse
import os
import numpy as np
from functools import reduce
from time import time



parser = argparse.ArgumentParser()  ### 1
parser.add_argument("--data_file", help=" training data file ", type=str) ### 2.1
parser.add_argument("--word2vec", help=" word2vec file ", type=str) ### 2.2
args = parser.parse_args()  ### 3
lines = open(args.data_file, 'r').readlines()
lines = lines[1:]
lines = map(lambda x:x.split('\t')[2], lines)

f1 = lambda x:int(x.split()[0])
f2 = lambda x:np.array([float(i) for i in x.split()[1:]])
word2vec = open(args.word2vec, 'r').readlines()[1:]
word2vec = {f1(line):f2(line) for line in word2vec}

print(word2vec[1])





