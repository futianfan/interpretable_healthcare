'''

python ./src/neural_net.py data/training_data_1.txt ./data/snow.Y $n ./data/test_data_1.txt

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
num = int(sys.argv[3])
###############################################################################################################
################################################################################################################
print('STEP 2. read-in data, read-in rule, assign data to rule')
##### input:   data/tmp3, results/corels_rule_list
##### output:  dict{rule: data-sample}
# read-in data:   data/tmp3
f3 = lambda x:[int(i) for i in x.rstrip().split()]
with open(sys.argv[1], 'r') as raw_data:
	data_lines = raw_data.readlines()
	data_dict = data_lines[1:]
# read-in rule:   results/corels_rule_list
f4 = lambda x: True if 'if ' in x and ' then ' in x else False
def rule2num(string):
	bgn = string.find('_') + 1
	endn = string.find('=')
	fea_num = int(string[bgn:endn])
	lab = 1 if 'yes' else 0
	return fea_num,lab

with open(sys.argv[2], 'r') as f_label:
	label_line = f_label.readlines()
	label = []
	for line in label_line:
		label.append(int(line))
label = np.array(label)

small_label = []
small_datadict = []
for i in range(len(label)):
	u = np.random.rand()
	if label[i] == 0 and u > 0.157:
		continue 
	small_label.append(label[i])
	small_datadict.append(data_dict[i])
small_label = np.array(small_label)

f5 = lambda x: 1 if x=='True' else 0
with open(sys.argv[4], 'r') as f_test:
	lines = f_test.readlines()
	test_data_dict = lines[1:]
	test_label = []
	for line in test_data_dict:
		test_label.append(f5(line.split()[0]))
	test_label = np.array(test_label)

#print(rule_dict[0])
################################################################################################################
print('NN training')

INPUT_SIZE = num ## embedding size => one-hot
HIDDEN_SIZE = 50 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 200
BATCH_SIZE = 256
#PROTOTYPE_NUM = len(rule_dict)
PROTOTYPE_NUM = 0
KERNEL_SIZE = 10
STRIDE = 3
CNN_OUT_SIZE = int((MAX_LENGTH - KERNEL_SIZE)/STRIDE) + 1
OUT_CHANNEL = 3
def data2array(data_dict):
	leng = len(data_dict)
	arr = np.zeros((leng, MAX_LENGTH, INPUT_SIZE),dtype = np.float32)
	arr_len = []
	for i in range(leng):
		line = data_dict[i]
		line = line.split('\t')
		admission = line[2].split()
		timestamp = line[3].split()
		assert len(admission) == len(timestamp)
		op_data = int(line[4])
		admission = admission[-MAX_LENGTH:]
		timestamp = timestamp[-MAX_LENGTH:]
		for j in range(len(admission)):
			adm = int(admission[j])
			arr[i,j,adm] = 1
			arr[i,j,-1] = op_data - int(timestamp[j]) 
		arr_len.append(len(admission))
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
        #self.conv1 = nn.Conv1d(in_channels = HIDDEN_SIZE, out_channels = OUT_CHANNEL, kernel_size = KERNEL_SIZE, stride = STRIDE)      
        #self.out1 = nn.Linear(HIDDEN_SIZE, OUT_SIZE)
        #self.out2 = nn.Linear(OUT_SIZE, 2)
        self.out3 = nn.Linear(HIDDEN_SIZE,2)

    def forward_rnn(self, X_batch, X_len):
    	batch_size = X_batch.shape[0]
    	#X_batch = Variable(torch.from_numpy(X_batch).float())
    	dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):  dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)
        X_out2 = X_out2[dd]
    	return X_out2  ## batch_size, HIDDEN_SIZE

    def forward(self, X_batch, X_len):
    	X_batch = torch.from_numpy(X_batch).float()
    	X_batch = Variable(X_batch)
    	batch_size = X_batch.shape[0]
    	X_out2 = self.forward_rnn(X_batch, X_len)
        ##X_out5 = F.relu(self.out1(X_out2))
        ##X_out6 = F.softmax(self.out2(X_out5))
        X_out6 = F.softmax(self.out3(X_out2))
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

LR = 4e-3 ### 1e-2 >> 1e-3, 1e-1 < 1e-2
nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True)
print('Build network')
opt_ = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

N = len(small_datadict)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
EPOCH = 1000
lamb = 1
for epoch in range(EPOCH):
    loss_average = 0        
    if epoch > 0 and epoch % 2 == 0:
    	t1 = time()
    	print('test: ', end = ' ')
    	#test_X(nnet, data_dict, label, epoch)
    	test_X(nnet, test_data_dict, test_label, epoch)
    	t2 = time()
    	print(str(t2-t1) + ' sec. ')
    t1 = time()
    for i in range(iter_in_epoch):
        ##### train
        #print('Epoch ' + str(epoch)+ ': iter ' + str(i) , end = ' ')
        t11 = time()
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x, batch_len = data2array(small_datadict[stt:endn])
        batch_label = small_label[stt:endn]
        batch_label = torch.from_numpy(batch_label)
        batch_label = Variable(batch_label)
        output = nnet(batch_x, batch_len)
        loss = loss_crossentropy(output, batch_label)
        opt_.zero_grad()
        #loss.backward(retain_variables = True)
        loss.backward()
        opt_.step()
        loss_value = loss.data[0]
        loss_average += loss_value
        #print('loss is ' + str(loss_value), end = ' ')
        t22 = time()
        #print(str(t22 - t11) + ' seconds')
        #print('Epoch: ' + str(epoch) + ", " + str(i) + "/"+ str(iter_in_epoch)+ ': loss value is ' + str(loss.data[0]))
        ##### train
    l_his.append(loss_average)
    t2 = time()
    print('Epoch '+str(epoch) + ': ' + str(int(t2-t1)) + ' sec. loss: ' + str(loss_average))
'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''




