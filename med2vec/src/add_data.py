'''
python2 ./src/add_data.py $INPUT_FILE results/rule_data_list results/similarity tmp 3
'''

import sys
train_data = open(sys.argv[1], 'r')
rule_data = open(sys.argv[2], 'r')
rule_similarity = open(sys.argv[3], 'r')
new_train_data = open(sys.argv[4], 'w')

rule_line = rule_data.readlines()
rule_index = [[int(i) for i in rule.split()] for rule in rule_line]
rule_len = [len(i) for i in rule_index]
## print(rule_len)

line = rule_similarity.readline()
rule_similarity = [float(i) for i in line.split()]
## print(rule_similarity)

n = len(rule_similarity)

similarity_threshold = 5000
rule_len_threshold = 100 

func = lambda i: rule_len[i] < rule_len_threshold and rule_similarity[i] < similarity_threshold
indx = filter(func,[i for i in range(n)])
print(indx)

indx = [rule_index[i] for i in indx]

#for i in indx:
#	print(len(i))
#for i in rule_index:
#	print(len(i))

origin_train_line = train_data.readlines()
new_train_data.write(''.join(origin_train_line))

new_add_data = [[origin_train_line[j + 1] for j in i] for i in indx]
new_add_data = [ ''.join(i) for i in new_add_data ]
new_add_data = ''.join(new_add_data)

repeat_times = int(sys.argv[5])
for i in range(repeat_times):
	new_train_data.write(new_add_data)



