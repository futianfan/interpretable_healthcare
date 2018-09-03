
# python generate_X_using_feature_selection.py $n tmp snow.X_new

import sys
n_dim = int(sys.argv[1])
reduce_dim = 50
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

Y_file = open(sys.argv[2], 'r')
lines = Y_file.readlines()
n_sample = len(lines)
Y = np.zeros((n_sample,))
for i, line in enumerate(lines):
	Y[i] = int(line)

X = np.zeros((n_sample, n_dim),dtype = np.float32)

with open(sys.argv[3], 'r') as fin:
	lines = fin.readlines()
	for indx, line in enumerate(lines):
		line = line.split()
		line = [int(i) for i in line]
		for i in range(n_dim):
			if i in line:
				X[indx,i] = 1


X_new = SelectKBest(chi2, k=reduce_dim).fit_transform(X, Y)  ## chi2, f_classif, mutual_info_classif
print(X_new.shape)

index = []
for i in range(reduce_dim):
	col = X_new[:,i]
	try:
		begin_num = index[-1]
	except:
		begin_num = 0
	for j in range(begin_num, n_dim):
		if (col == X[:,j]).all():
			index.append(j)
			break 

fout = open(sys.argv[4], 'w')
for i in range(reduce_dim):
	## yes  single rule
	feature_name = '{F_' + str(index[i]) + '=yes} '
	str_all = ['1' if X_new[j,i] == 1 else '0' for j in range(n_sample)]
	str_all = ' '.join(str_all) + '\n'
	fout.write(feature_name + str_all)
	#####################################################
	## no
	feature_name = '{F_' + str(index[i]) + '=yes} '
	str_all = ['0' if X_new[j,i] == 1 else '1' for j in range(n_sample)]
	str_all = ' '.join(str_all) + '\n'
	fout.write(feature_name + str_all)
	################################################################################
	## double 	
	for j in range(i + 1, reduce_dim):
		feature_name = '{F_' + str(index[i]) + '=yes,'+ 'F_' + str(index[j]) +'=yes} '
		str_all = ['1' if X_new[k,i] == 1 and X_new[k,j] == 1 else '0' for k in range(n_sample)]
		str_all = ' '.join(str_all) + '\n'
		fout.write(feature_name + str_all)
	################################################################################
fout.close()






