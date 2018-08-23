'''
python ./src/find_key_factor.py data/tmp3 50 data/key_factor
'''
import sys
filename = sys.argv[1]
num = int(sys.argv[2])
fileout = sys.argv[3]  ### data/key_factor
fin = open(filename, 'r')
lines = fin.readlines()
fin.close()
dic = dict()
N = 1870
count_lst = [0 for i in range(N)]
for line in lines:
	line = [int(i) for i in line.split()]
	indx_set = set(line)
	for i in indx_set: count_lst[i] += 1

count_lst2 = count_lst[:]
count_lst2.sort()
threshold = count_lst2[-num]	
f = lambda x: count_lst[x] >= threshold
f2 = lambda x: count_lst[x] > threshold 
indx = filter(f, [i for i in range(N)])
fout = open(fileout, 'w')
for i in indx: fout.write(str(i) + ' ')
fout.close()




