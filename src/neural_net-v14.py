'''

python ./src/neural_net.py data/training_model_by_word2vec_1.vector data/tmp3 \
results/corels_rule_list ./data/snow.Y ./data/test_data_1_3.txt ./data/test_snow.Y \
./data/train_lstm_output.npy ./data/test_lstm_output.npy ./results/rule_data_list

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
from sklearn.metrics import roc_auc_score
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

with open(sys.argv[4], 'r') as f_label:
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
with open(sys.argv[5], 'r') as f_test:
    lines = f_test.readlines()
    test_data_dict = [f3(line) for i,line in enumerate(lines)]

with open(sys.argv[6], 'r') as f_label:
    label_line = f_label.readlines()
    test_label = []
    for line in label_line:
        test_label.append(int(line))
test_label = np.array(test_label)


# read-in rule:   results/corels_rule_list
f4 = lambda x: True if 'if ' in x and ' then ' in x else False
def rule2num(string):
	bgn = string.find('_') + 1
	endn = string.find('=')
	fea_num = int(string[bgn:endn])
	lab = 1 if 'yes' else 0
	return fea_num,lab

## assign data to rule
f_rule_out = open(sys.argv[9], 'w')
print(len(data_dict))
with open(sys.argv[3], 'r') as f_rule:  ## 'results/corels_rule_list'    
    lines = f_rule.readlines()
    lines = filter(f4,lines)
    rule_dict = {}
    for indx, line in enumerate(lines):
        bgn = line.find('({')
        endn = line.find('})')
        rule = line[bgn + 2:endn]
        dic = {rule2num(j)[0]:rule2num(j)[1] for j in rule.split(',')}  # 1200:1, 398:1 
        #print(indx)
        def rule_match_data(data_index):
            line = data_dict[data_index]
            for fea_num,lab in dic.items():
                if fea_num not in line and lab == 1:
                    return False
                if fea_num in line and lab == 0:
                    return False
            return True
        rule_dict[indx] = filter(rule_match_data, range(len(data_dict)))  ### data_dict.keys()
        string = [str(i) for i in rule_dict[indx]]
        string = ' '.join(string)
        f_rule_out.write(string + '\n')
        print(len(rule_dict[indx]), end = ' ')
print('num of matching data for each rule')
f_rule_out.close()
print(len(rule_dict))



#print(rule_dict[0])
################################################################################################################
print('NN training')

INPUT_SIZE = 100 ## embedding size
HIDDEN_SIZE = 30 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 50
BATCH_SIZE = 256
PROTOTYPE_NUM = len(rule_dict)

KERNEL_SIZE = 10
STRIDE = 1
CNN_OUT_SIZE = int((MAX_LENGTH - KERNEL_SIZE)/STRIDE) + 1
OUT_CHANNEL = INPUT_SIZE
MAXPOOL_NUM = 3
INPUT_SIZE_RNN = OUT_CHANNEL
NUM_HIGH_WAY = 1
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
                #print(line[j], end='** ')
                pass
        arr_len.append(len(line))
    return arr, arr_len  ### batch size, MAX_LENGTH, INPUT_SIZE
'''
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
            try:
                arr[i,j,:] = id2vec[str(line[j])]
            except:
                print(str(line[j]),end = '** ')
        arr_len.append(len(admission))
    return arr, arr_len
'''


class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   
        self.rnn1 = nn.LSTM(
            input_size = INPUT_SIZE_RNN, 
            hidden_size = HIDDEN_SIZE / 2,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True
            )      
        self.out1 = nn.Linear(PROTOTYPE_NUM, OUT_SIZE)
        self.out2 = nn.Linear(OUT_SIZE, 2)
    	## PROTOTYPE_NUM, HIDDEN_SIZE 
    	self.prototype = torch.zeros(PROTOTYPE_NUM, HIDDEN_SIZE)
    	self.prototype = Variable(self.prototype)

        ####  CNN 
        self.out3 = nn.Linear(HIDDEN_SIZE,2)
        self.conv1 = nn.Conv1d(in_channels = INPUT_SIZE, out_channels = OUT_CHANNEL, kernel_size = KERNEL_SIZE, stride = STRIDE)   
        self.maxpool = nn.MaxPool1d(kernel_size = MAXPOOL_NUM)

        #### highway
        self.num_layers = NUM_HIGH_WAY
        self.nonlinear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.linear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.gate = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.f = F.relu 

    def forward_highway(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

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

        ### Option I
        #X_out, _ = self.rnn1(pack_X_batch, None) 
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]

        ### Option II
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)

        X_out2 = X_out2[dd]    ## batch_size, HIDDEN_SIZE
    	return X_out2

    def forward_A(self, X_batch, X_len):
    	X_batch = torch.from_numpy(X_batch).float()
    	X_batch = Variable(X_batch)
    	batch_size = X_batch.shape[0]

        ### cnn + rnn
        X_batch_2 = X_batch.permute(0,2,1)  ## batch size, INPUT_SIZE, MAX_LENGTH
        X_batch_3 = self.conv1(X_batch_2)
        X_batch_4 = self.maxpool(X_batch_3)
        f_map = lambda x: max(int((int((x - KERNEL_SIZE) / STRIDE) + 1) / MAXPOOL_NUM),1)
        X_len2 = map(f_map, X_len)
        X_batch_4 = X_batch_4.permute(0,2,1)
        #print(X_batch_4.shape)
        #print(X_len)
        #print(X_len2)

        X_out2 = self.forward_rnn(X_batch_4, X_len2)
        return X_out2 
    	#X_out2 = self.forward_rnn(X_batch, X_len)
        
        ##X_out2 = self.forward_highway(X_out2)
        ## highway 

        ### full connected


    def forward(self, X_batch, X_len):
        X_out2 = self.forward_A(X_batch, X_len)
        X_out6 = F.softmax(self.out3(X_out2))
        return X_out6
        '''
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
        matching_loss, _ = torch.min(X_out4,1)
        matching_loss = torch.mean(matching_loss)
        return X_out6, matching_loss
        '''
    def generate_prototype(self, data_dict, rule_dict):
    	for i in range(len(rule_dict)):
    		data = []
    		for j in rule_dict[i]:
    			data.append(data_dict[j])
            #data = [data_dict[j] for j in rule_dict[i]]
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
    #fout = open('./results/test_result_of_epoch_' + str(epoch), 'w')
    assert len(data_dict) == len(data_label)
    iter_num = int(np.ceil(N_test * 1.0 / BATCH_SIZE))
    y_pred = []
    y_label = []
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
            #fout.write(str(data_label[stt + j]) + ' ' + str(output_data[j][0]) + '\n')
            y_pred.append(output_data[j][1])
            y_label.append(data_label[stt+j])
    auc = roc_auc_score(y_label, y_pred)
    print('AUC of Epoch ' + str(epoch) + ' is ' + str(auc)[:5])

def save_lstm_output_for_testdata(nnet, data_dict, data_label, fname):
    ## data_dict, data_label is a list, [0,1,1,1,0 0 0 0 ]
    N_test = len(data_dict)
    #fout = open('./results/test_result_of_epoch_' + str(epoch), 'w')
    assert len(data_dict) == len(data_label)
    iter_num = int(np.ceil(N_test * 1.0 / BATCH_SIZE))
    for i in range(iter_num):
        stt = i * BATCH_SIZE
        endn = min(N_test, stt + BATCH_SIZE)
        batch_x, batch_len = data2array(data_dict[stt:endn])
        if batch_x.shape[0] == 0:
            break
        output = nnet.forward_A(batch_x, batch_len)
        if i == 0:
            output2 = output 
        else: 
            output2 = torch.cat((output2, output), 0)
    output2 = output2.data 
    output2 = np.array(output2)
    np.save(fname, output2)

LR = 1e-1 ### 4e-3 is ok,
EPOCH = 39

nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True)
print('Build network')
nnet.generate_prototype(data_dict, rule_dict)
print('Generate prototype')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

N = len(data_dict)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
lamb = 4e-3
for epoch in range(EPOCH):
    if epoch > 6 and epoch % 3 == 0:
        t1 = time()
        #print('test: ', end = ' ')
        #test_X(nnet, data_dict, label, epoch)
        test_X(nnet, test_data_dict, test_label, epoch)
        t2 = time()
        #print(str(int(t2-t1)) + ' sec. ')
    loss_average = 0
    t1 = time()
    for i in range(iter_in_epoch):
        ##### train
        #print('Epoch ' + str(epoch)+ ': iter ' + str(i) , end = ' ')
        t11 = time()
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x , batch_len = data2array(data_dict[stt:endn])
        batch_label = label[stt:endn]
        batch_label = torch.from_numpy(batch_label)
        batch_label = Variable(batch_label)
        output = nnet(batch_x, batch_len)
        loss = loss_crossentropy(output, batch_label)
        opt_.zero_grad()
        loss.backward()  ##retain_variables = True
        opt_.step()
        loss_value = loss.data[0]
        loss_average += loss_value
        #print('loss is ' + str(loss_value)[:7], end = ' ')
        t22 = time()
        #print(str((t22 - t11))[:5] + ' seconds')
        #print('Epoch: ' + str(epoch) + ", " + str(i) + "/"+ str(iter_in_epoch)+ ': loss value is ' + str(loss.data[0]))
        ##### train
    l_his.append(loss_average)
    t2 = time()
    print('Epoch '+str(epoch) + ' takes ' + str(int(t2-t1)) + ' sec. loss: ' + str(loss_average)[:6])

###  save  train data
for i in range(iter_in_epoch):
    stt = i * BATCH_SIZE
    endn = min(N,stt + BATCH_SIZE)
    batch_x , batch_len = data2array(data_dict[stt:endn])
    output = nnet.forward_A(batch_x , batch_len)
    if i == 0:
        output2 = output 
    else: 
        output2 = torch.cat((output2, output), 0)
output2 = output2.data 
output2 = np.array(output2)
np.save(sys.argv[7], output2)
###  save  train data
######################################################
###### save  test data
save_lstm_output_for_testdata(nnet, test_data_dict, test_label, sys.argv[8])
###### save  test data


'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''






