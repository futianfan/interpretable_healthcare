import sys
max_i = 0

with open(sys.argv[1] , 'r') as fin:
	while True:
		line = fin.readline()
		if line == '':
			break
		line = line.split()
		line = [int(i) for i in line]
		if max(line) > max_i:
			max_i = max(line)

print(str(max_i))




