''' 
python src/union_rule.py --origin_rule_file results/rule_data_list --united_rule_file results/union_rule_data_list --union_num 15 
''' 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--origin_rule_file", help=" input rule file ", type=str)
parser.add_argument("--united_rule_file", help="output rule file", type=str)
parser.add_argument("--union_num", help="union's number", type=str)
args = parser.parse_args()

input_file = open(args.origin_rule_file, 'r')
output_file = open(args.united_rule_file, 'w')
num = int(args.union_num)

lines = input_file.readlines()
input_file.close()
leng = len(lines)
leng_o = leng // num 

for i in range(leng_o):
	if i < leng_o - 1:
		multiple_lines = [line.rstrip() for line in lines[i*num:i*num+num]]
	else:
		multiple_lines = [line.rstrip() for line in lines[i * num:]]
	multiple_lines = ' '.join(multiple_lines) + '\n'
	output_file.write(multiple_lines)

output_file.close()



