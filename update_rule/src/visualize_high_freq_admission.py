'''
python2 src/visualize_high_freq_admission.py --train_data 
'''


import sys
import argparse
import os
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser()
parser.add_argument('--train_data', help = 'train data file', type = str)
parser.add_argument('--rule_assignment', help = 'rule assignment file', type = str)


