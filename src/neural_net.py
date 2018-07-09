## python ./src/neural_net.py data/id2vec.txt data/tmp3 results/corels_rule_list

from __future__ import print_function
import sys
import numpy as np 

######################################################## 
print('STEP 1. read-in id2vec')
f1 = lambda x:x.split()[0]
f2 = lambda x:np.array([float(i) for i in x.split()[1:]])
with open(sys.argv[1], 'r') as word2vec_file:
	lines = word2vec_file.readlines()
	lines = lines[1:]
	id2vec = {f1(line):f2(line) for line in lines}

########################################################
print('STEP 2. read-in data, read-in rule, assign data to rule')
## read-in data:   data/tmp3
f3 = lambda x:[int(i) for i in x.rstrip().split()]
with open(sys.argv[2], 'r') as raw_data:
	lines = raw_data.readlines()
	data_dict = {i:f3(line) for i,line in enumerate(lines)}
## read-in rule:   results/corels_rule_list
f4 = lambda x: True if 'if ' in x and ' then ' in x else False
def rule2num(string):
	bgn = string.find('_') + 1
	endn = string.find('=')
	fea_num = int(string[bgn:endn])
	lab = 1 if 'yes' else 0
	return fea_num,lab

with open(sys.argv[3], 'r') as f_rule:
	lines = f_rule.readlines()
	lines = filter(f4,lines)
	rule_dict = {}
	for indx, line in enumerate(lines):
		bgn = line.find('({')
		endn = line.find('})')
		rule = line[bgn + 2:endn]
		dic = {rule2num(j)[0]:rule2num(j)[1] for j in rule.split(',')}
		rule_dict[indx] = dic




## assign data to rule






