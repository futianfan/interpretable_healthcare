'''
python2 ./src/pure_rule_learning.py --rule_file ./corels/rule_list_result --test_file ./data/training_data_1.txt 
python2 ./src/pure_rule_learning.py --rule_file ./corels/rule_list_result --test_file ./data/test_data_1.txt 
'''

import sys
import argparse
import os
import numpy as np
from time import time
from sklearn.metrics import roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument("--test_file", help=" test data file ", type=str)
parser.add_argument("--rule_file", help="rule file", type=str)
args = parser.parse_args()

test_file = open(args.test_file, 'r')
rule_file = open(args.rule_file, 'r')



###########################################################################
### 					rule
###########################################################################
lines = rule_file.readlines()
f1 = lambda x:lines[x][:4] == "if ("
f2 = lambda x:lines[x][:6] == "else ("
leng = len(lines)
stt = filter(f1, list(range(leng)))
endn = filter(f2, list(range(leng)))
stt = list(stt)[0]
endn = list(endn)[0]
rule_list = lines[stt:endn+1]
leng_rule = len(rule_list)
rule_key_list = []
rule_label_list = []
f3 = lambda x: 1 if 'Yes' in x else 0
f4 = lambda x: 1 if 'yes' in x else 0
def rulekey2dict(rule_key):
	token = rule_key.split(',')
	leng = len(token)
	dic = {}
	for i in token:
		label = f4(i)
		stt = i.index('_')
		endn = i.index('=')
		num = int(i[stt+1:endn])
		dic[num] = label
	return dic 


for i in range(leng_rule - 1):
	rule = rule_list[i]
	indx = rule.index('if ') + 5 
	indx2 = rule.index('}')
	rule_key = rule[indx:indx2] 
	##print(rule_key)
	dic = rulekey2dict(rule_key)
	##print(dic)
	rule_key_list.append(dic)
	indx3 = rule[indx2+1:].index('{')
	indx4 = rule[indx2+1:].index('}')
	rule_label = rule[indx2+1:][indx3+1:indx4]
	label = f3(rule_label)
	rule_label_list.append(label)

####
rule = rule_list[-1]
assert rule[:7] == 'else ({'
indx1 = rule.index('{')
indx2 = rule.index('}')
rule_label = rule[indx1+1:indx2]
label = f3(rule_label)
rule_label_list.append(label)
rule_key_list.append(dict())
####
print(rule_key_list)
print(len(rule_label_list))
###########################################################################


###########################################################################
####					test data
###########################################################################
lines = test_file.readlines()[1:]
f1 = lambda x:1 if 'True' in x else 0 
data_label = []
data_seqs = []
for line in lines:
	tokens = line.rstrip().split('\t')
	data_label.append(f1(tokens[0]))
	seqs = tokens[2].split()
	seqs = [int(i) for i in seqs]
	data_seqs.append(seqs)


def judge_a_rule_a_seq(rule, seqs):
	if len(rule) == 0:
		return True
	for i in rule:
		ii = i
		if (rule[i] == 1 and ii not in seqs) or (rule[i] == 0 and ii in seqs):  ###*****************
			return False
	return True


def judge_multiple_rule_a_seq(rule_key_lists, rule_label_lists, seqs):
	for i, j in enumerate(rule_key_lists):
		if judge_a_rule_a_seq(j, seqs):
			return rule_label_lists[i]


data_predict_label = []
for i, seq in enumerate(data_seqs):
	if i % 10 == 0: print(i)
	data_predict_label.append(judge_multiple_rule_a_seq(rule_key_list, rule_label_list, seq))

print(data_predict_label)
print(sum(data_predict_label))
auc_value = roc_auc_score(data_label, data_predict_label)
print(auc_value)











