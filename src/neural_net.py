'''
python ./src/neural_net.py data/id2vec.txt data/tmp3 results/corels_rule_list
'''

from __future__ import print_function
import sys
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
torch.manual_seed(1)    # reproducible
################################################################################################################ 
print('STEP 1. read-in id2vec')
f1 = lambda x:x.split()[0]
f2 = lambda x:np.array([float(i) for i in x.split()[1:]])
with open(sys.argv[1], 'r') as word2vec_file:
	lines = word2vec_file.readlines()
	lines = lines[1:]
	id2vec = {f1(line):f2(line) for line in lines}

################################################################################################################
print('STEP 2. read-in data, read-in rule, assign data to rule')
##### input:   data/tmp3, results/corels_rule_list
##### output:  dict{rule: data-sample}
# read-in data:   data/tmp3
f3 = lambda x:[int(i) for i in x.rstrip().split()]
with open(sys.argv[2], 'r') as raw_data:
	data_lines = raw_data.readlines()
	data_dict = {i:f3(line) for i,line in enumerate(data_lines)}
# read-in rule:   results/corels_rule_list
f4 = lambda x: True if 'if ' in x and ' then ' in x else False
def rule2num(string):
	bgn = string.find('_') + 1
	endn = string.find('=')
	fea_num = int(string[bgn:endn])
	lab = 1 if 'yes' else 0
	return fea_num,lab

## assign data to rule
with open(sys.argv[3], 'r') as f_rule:
	lines = f_rule.readlines()
	lines = filter(f4,lines)
	rule_dict = {}
	for indx, line in enumerate(lines):
		bgn = line.find('({')
		endn = line.find('})')
		rule = line[bgn + 2:endn]
		dic = {rule2num(j)[0]:rule2num(j)[1] for j in rule.split(',')}  # 1200:1, 398:1 
		def rule_match_data(data_index):
			line = data_dict[data_index]
			for fea_num,lab in dic.items():
				if fea_num not in line and lab == 1:
					return False
				if fea_num in line and lab == 0:
					return False
			return True
		rule_dict[indx] = filter(rule_match_data, data_dict.keys())
		print(len(rule_dict[indx]), end = ' ')
#print(rule_dict[0])
################################################################################################################
print('NN training')

INPUT_SIZE = 100 ## embedding size
HIDDEN_SIZE = 30 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 200
BATCH_SIZE = 32

def data2array(data_dict):
	leng = len(data_dict)
	arr = np.zeros((leng, MAX_LENGTH, INPUT_SIZE),dtype = np.float32)
	for i in range(leng):
		line = data_dict[i]
		line = line[-MAX_LENGTH:]
		for j in range(len(line)):
			arr[i,j,:] = id2vec[line[j]]
	return arr 

class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   
        self.rnn1 = nn.LSTM(
            input_size = INPUT_SIZE, 
            hidden_size = HIDDEN_SIZE,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST
            )      
        self.out1 = nn.Linear(PROTOTYPE_NUM, OUT_SIZE)
        self.out2 = nn.Linear(OUT_SIZE, 2)

    	prot = np.zeros((PROTOTYPE_NUM,HIDDEN_SIZE),dtype = np.float32)		## PROTOTYPE_NUM, HIDDEN_SIZE 
    	self.prototype = torch.from_numpy(prot).float()

'''    def generate_prototype(self, data_dict, rule_dict):
    	for i in range(len(rule_dict)):
    		data = []
    		for j in rule_dict[i]:
    			data.append(data_dict[j])
    		leng = len(data)




    		self.prototype[i,:] = 




    	self.prototype = Variable(self.prototype)
    	return
'''
    def forward(self, X_batch, X_len):
    	batch_size = X_batch.shape[0]
    	dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch = torch.from_numpy(X_batch).float()
        X_batch_v = Variable(X_batch)
        X_batch_v = X_batch_v[dd1]
        #X_batch_v = X_batch_v.cuda()
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        X_out, _ = self.rnn1(pack_X_batch, None)
        unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        indx = list(np.array(X_len_sort) - 1)
        indx = [int(v) for v in indx]
        X_out2 = unpack_X_out[range(batch_size), indx]
        X_out2 = X_out2[dd]  ## batch_size, HIDDEN_SIZE 

        ###### prototype
        X_out2 = X_out2.view(batch_size,HIDDEN_SIZE,1)
        X_out3 = X_out2.expand(batch_size, HIDDEN_SIZE, PROTOTYPE_NUM)
        prtt = self.prototype.view(1, HIDDEN_SIZE, PROTOTYPE_NUM)
        prtt = prtt.expand(batch_size, HIDDEN_SIZE, PROTOTYPE_NUM)
        X_diff = (X_out3 - prtt)**2
        X_out4 = torch.sum(X_diff, 1)  ### batch_size, PROTOTYPE_NUM
        ###### prototype
        X_out5 = F.relu(self.out1(X_out4))
        X_out6 = F.softmax(self.out2(X_out5))
        return X_out6


















