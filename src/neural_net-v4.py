'''
python ./src/neural_net.py data/id2vec.txt data/tmp3 results/corels_rule_list ./data/snow.Y
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
from time import time
torch.manual_seed(1)    # reproducible
np.random.seed(1)
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
	#data_dict = {i:f3(line) for i,line in enumerate(data_lines)}
	data_dict = [f3(line) for i,line in enumerate(data_lines)]
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
		rule_dict[indx] = filter(rule_match_data, range(len(data_dict)))  ### data_dict.keys()
		print(len(rule_dict[indx]), end = ' ')
print('num of matching data for each rule')
with open(sys.argv[4], 'r') as f_label:
	label_line = f_label.readlines()
	label = []
	for line in label_line:
		label.append(int(line))
label = np.array(label)
#print(rule_dict[0])
################################################################################################################
print('NN training')

INPUT_SIZE = 100 ## embedding size
HIDDEN_SIZE = 50 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 200
BATCH_SIZE = 256
PROTOTYPE_NUM = len(rule_dict)

def data2array(data_dict):
	leng = len(data_dict)
	arr = np.zeros((leng, MAX_LENGTH, INPUT_SIZE),dtype = np.float32)
	arr_len = []
	for i in range(leng):
		line = data_dict[i]
		line = line[-MAX_LENGTH:]
		for j in range(len(line)):
			try:
				arr[i,j,:] = id2vec[str(line[j])]
			except:
				pass 
		arr_len.append(len(line))
	return arr, arr_len

class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   
        self.rnn1 = nn.LSTM(
            input_size = INPUT_SIZE, 
            hidden_size = HIDDEN_SIZE / 2,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True
            )      
        self.out1 = nn.Linear(HIDDEN_SIZE, OUT_SIZE)
        self.out2 = nn.Linear(OUT_SIZE, 2)

    def forward_rnn(self, X_batch, X_len):
    	batch_size = X_batch.shape[0]
    	#X_batch = Variable(torch.from_numpy(X_batch).float())
    	dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        __, (X_out,_) = self.rnn1(pack_X_batch, None)
        #X_out2 = np.concatenate((X_out[0], X_out[1]), axis = 1)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]
        X_out2 = X_out2[dd]  ## batch_size, HIDDEN_SIZE 
    	return X_out2

    def forward(self, X_batch, X_len):
    	X_batch = torch.from_numpy(X_batch).float()
    	X_batch = Variable(X_batch)
    	batch_size = X_batch.shape[0]
    	X_out2 = self.forward_rnn(X_batch, X_len)
        ###### prototype
        X_out2 = X_out2.view(batch_size,HIDDEN_SIZE)
        X_out5 = F.relu(self.out1(X_out2))
        X_out6 = F.softmax(self.out2(X_out5))
        return X_out6

    def generate_prototype(self, data_dict, rule_dict):
    	for i in range(len(rule_dict)):
    		data = []
    		for j in rule_dict[i]:
    			data.append(data_dict[j])
    		leng = len(data)
    		T = np.ceil(leng * 1.0 / BATCH_SIZE)
    		T = int(T)
    		for j in range(T):
    			bgn = j * BATCH_SIZE
    			endn = min(leng, bgn + BATCH_SIZE)
    			batch_data, batch_len = data2array(data[bgn:endn])
    			batch_data = Variable(torch.from_numpy(batch_data).float())
    			batch_X_out = self.forward_rnn(batch_data, batch_len)
    			if j==0:
    				X_out = batch_X_out
    			else:
    				X_out = torch.cat([X_out, batch_X_out], 0)
    		self.prototype[i,:] = torch.mean(X_out, 0)
    	#self.prototype = self.prototype.data.numpy()
    	#self.prototype = torch.from_numpy(self.prototype).float()


def test_X(nnet, data_dict, data_label, epoch):
    ## data_dict, data_label is a list, [0,1,1,1,0 0 0 0 ]
    N_test = len(data_dict)
    fout = open('./results/test_result_of_epoch_' + str(epoch), 'w')
    assert len(data_dict) == len(data_label)
    iter_num = int(np.ceil(N_test * 1.0 / BATCH_SIZE))
    for i in range(iter_num):
    	stt = i * BATCH_SIZE
    	endn = min(N_test, stt + BATCH_SIZE)
    	batch_x, batch_len = data2array(data_dict[stt:endn])
    	if batch_x.shape[0] == 0:
    		break
    	output = nnet(batch_x, batch_len)
    	output_data = output.data 
    	for j in range(output_data.shape[0]):
    		#print(str(data_label[stt + j]) + ' ' + str(output_data[j][0]))
    		fout.write(str(data_label[stt + j]) + ' ' + str(output_data[j][0]) + '\n')

LR = 3e-3 ### 1e-2 >> 1e-3, 1e-1 < 1e-2
nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True)
print('Build network')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

N = len(data_dict)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
EPOCH = 256
lamb = 1
for epoch in range(EPOCH):
    loss_average = 0
    t1 = time()
    print('test: ', end = ' ')
    test_X(nnet, data_dict, label, epoch)
    t2 = time()
    print(str(t2-t1) + ' seconds. ')
    t1 = time()
    for i in range(iter_in_epoch):
        ##### train
        print('Epoch ' + str(epoch)+ ': iter ' + str(i) , end = ' ')
        t11 = time()
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x, batch_len = data2array(data_dict[stt:endn])
        batch_label = label[stt:endn]
        batch_label = torch.from_numpy(batch_label)
        batch_label = Variable(batch_label)
        output = nnet(batch_x, batch_len)
        loss = loss_crossentropy(output, batch_label)
        opt_.zero_grad()
        loss.backward(retain_variables = True)
        opt_.step()
        loss_value = loss.data[0]
        loss_average += loss_value
        print('loss is ' + str(loss_value), end = ' ')
        t22 = time()
        print(str(t22 - t11) + ' seconds')
        #print('Epoch: ' + str(epoch) + ", " + str(i) + "/"+ str(iter_in_epoch)+ ': loss value is ' + str(loss.data[0]))
        ##### train
    l_his.append(loss_average)
    t2 = time()
    print('Epoch '+str(epoch) + ' takes ' + str(t2-t1) + ' seconds')
'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''




