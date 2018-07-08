
# python generate_X_data.py $n tmp snow.X

import sys
num_of_word = int(sys.argv[1])


with open(sys.argv[2], 'r') as fin:
	with open(sys.argv[3], 'w') as fout:
		while True:
			line = fin.readline()
			if line == '':
				break
			line = [int(i) for i in line.split()]
			str_all = ['F_' + str(i) + '=yes' \
			if i in line else 'F_' + str(i) + '=no' \
			for i in range(num_of_word)]
			str_all = ' '.join(str_all)
			fout.write(str_all + '\n')
		



