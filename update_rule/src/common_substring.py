# -*- coding: utf-8 -*-
'''
python src/common_substring.py --data_file ./data/train_data --rule_data_index ./results/rule_data_list --dictionary ./data/SNOW_vocabMAP.txt
'''

import argparse
import os
import numpy as np
from functools import reduce
from time import time

def backtracking(mat_a, lst1, lst2):
	l1, l2 = mat_a.shape 
	i, j = l1 - 1, l2 - 1
	lcs = []
	while True:
		if i == 0 or j==0:
			return lcs 
		if lst1[i-1] == lst2[j-1]:
			lcs = [lst1[i-1]] + lcs
			return backtracking(mat_a[:-1,:-1], lst1, lst2) + lcs 
		elif mat_a[i, j-1] > mat_a[i-1, j]:
			j -= 1 
			return backtracking(mat_a[:,:-1], lst1, lst2)
		else:
			i -= 1
			return backtracking(mat_a[:-1,:], lst1, lst2)

def largest_common_substring(lst1, lst2):
	len1 = len(lst1)
	len2 = len(lst2)
	mat_a = np.zeros((len1 + 1, len2 + 1), dtype=int)
	flag = np.zeros((len1 + 1, len2 + 1), dtype=int)
	for i in range(1,len1+1):
		for j in range(1,len2+1):
			mat_a[i,j] = mat_a[i-1,j-1] + 1 if lst1[i-1] == lst2[j-1] else max(mat_a[i-1,j], mat_a[i,j-1])
	lcs = backtracking(mat_a, lst1, lst2)
	return mat_a[len1, len2], lcs 

'''
a = [1,2,3,4,5,6,7]
b = [2,4,3,6]
## 3 2,3,6 or 2,4,6
#a='ABCBDAB'
#b='BDCABA'
## 4 BCBA
print(largest_common_substring(a,b))
exit()
'''


def remove_repetition(lst):
	leng = len(lst)
	lst2 = []
	for i in range(leng):
		if lst[i] not in lst[:i]:
			lst2.append(lst[i])
	return lst2 

def largest_common_substring_in_a_cluster(data):
	leng = len(data)
	X = data[0]
	for i in range(1,leng):
		##print(i, end=',')
		_, X = largest_common_substring(X, data[i])
	return X 


def code2name(dic, lst):
	lst =  [dic[i] for i in lst]
	return ' -> '.join(lst)

## print(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--data_file", help=" training data file ", type=str)
parser.add_argument("--rule_data_index", help="rule data index file", type=str)
parser.add_argument("--dictionary", help="dictionary: code => diagnois", type=str)
args = parser.parse_args()

fp_data = open(args.data_file, 'r')
fp_rule = open(args.rule_data_index, 'r')
dictionary = open(args.dictionary, 'r')

lines = fp_data.readlines()
lines = lines[1:]
lines = [[int(i) for i in line.rstrip().split('\t')[2].split()] for line in lines]

rules = fp_rule.readlines()
rules = [[int(i) for i in rule.split()] for rule in rules]
num_of_rules = len(rules)

dict_line = dictionary.readlines()
dic = {int(line.rstrip().split('\t')[0]):line.rstrip().split('\t')[1] for line in dict_line}
#print(dic[1799])
#exit()

'''
for rule_num in range(num_of_rules):
	rule = rules[rule_num]
	data = [lines[i] for i in rule]
	data2 = remove_repetition(data)
	leng = len(data2)
	if leng == 1:
		continue 
	print(leng)
	t1 = time()
	matching_leng = []
	for i in range(leng):
		for j in range(i + 1, leng):
			lcs_len, lcs_string = largest_common_substring(data2[i], data2[j])
			##print(lcs, end= ', ')
			matching_leng.append(lcs_len)
			print(lcs_string)
	max_leng = reduce(max, matching_leng)
	print(str(rule_num) + ': ' + str(leng) + ' points, max matching lens is ' + str(max_leng) + ', costs ' + str(time() - t1)[:5] + ' sec')
'''


'''
for rule_num in range(num_of_rules):
	rule = rules[rule_num]
	data = [lines[i] for i in rule]
	data2 = remove_repetition(data)
	leng = len(data2)
	if leng == 1:
		continue 
	print(leng)
	t1 = time()
	matching_leng = []
	for i in range(leng):
		for j in range(i + 1, leng):
			lcs_len, lcs_string = largest_common_substring(data2[i], data2[j])
			##print(lcs, end= ', ')
			matching_leng.append(lcs_len)
			#print(lcs_string)
	max_leng = reduce(max, matching_leng)
	print(str(rule_num) + ': ' + str(leng) + ' points, max matching lens is ' + str(max_leng) + ', costs ' + str(time() - t1)[:5] + ' sec')
'''




for rule_num in range(num_of_rules):
	rule = rules[rule_num]
	data = [lines[i] for i in rule]
	data2 = remove_repetition(data)
	leng = len(data2)
	if leng == 1:
		continue 
	t1 = time()
	X = largest_common_substring_in_a_cluster(data2)
	print(str(rule_num) + '-th rule:' + str(leng) + ' points, max-common lens:' + str(len(X)))
	##print(str(rule_num) + '-th rule:' + str(leng) + ' points, max-common lens:' + str(len(X)) + ', cost ' + str(time() - t1)[:5] + 'sec')


## 2153,4268,9458,10579,10819
## sed -n "10821p" data/training_data_1.txt | awk '{print $1}'
rule = rules[19]
print(rule)
data = [lines[i] for i in rule]
data2 = remove_repetition(data)
X = largest_common_substring_in_a_cluster(data2)
print(code2name(dic, X))
for i in data2:
	print(code2name(dic, i))






