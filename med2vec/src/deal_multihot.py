'''
python2 ./src/deal_multihot.py --train_file ./data/train_data --multihot_train_data ./data/multihot_train_data
'''

import sys
import argparse
import os
import numpy as np
from time import time


parser = argparse.ArgumentParser()
parser.add_argument("--train_file", help=" train file ", type=str)
parser.add_argument("--multihot_train_data", help="multihot train file ", type=str)
args = parser.parse_args()


fin = open(args.train_file, 'r')
lines = fin.readlines()[1:]

def line_2_list_of_line(line):
	admission = line.split('\t')[2].split()
	timestamp = line.split('\t')[3].split()
	leng = len(admission)
	assert leng == len(timestamp)

	current_time = timestamp[0]
	big_lst = ''
	small_lst = admission[0]
	for i in range(1,leng):
		if timestamp[i] == current_time:
			small_lst += ' ' + admission[i]
		else:
			big_lst += small_lst + '\t'
			small_lst = admission[i]
			current_time = timestamp[i]
	big_lst += small_lst + '\n'
	return big_lst

fout = open(args.multihot_train_data, 'w')
new_lines = map(line_2_list_of_line, lines)
for line in new_lines:
	fout.write(line)
fout.close()



