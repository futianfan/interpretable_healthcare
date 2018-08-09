'''

python ./src/neural_net.py data/training_model_by_word2vec_1.vector data/tmp3 \
results/corels_rule_list ./data/snow.Y ./data/test_data_1_3.txt ./data/test_snow.Y \
./data/train_lstm_output.npy ./data/test_lstm_output.npy ./results/rule_data_list

'''

from __future__ import print_function
import sys
import torch
from torch.autograd import Variable
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 
from time import time
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)
Num = 1865
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

################################################################################################################
print('NN training')

INPUT_SIZE = 100 ## embedding size
HIDDEN_SIZE = 20 
OUT_SIZE = 10
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 15
BATCH_SIZE = 100  ## 256
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


class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   
        self.encoder_rnn = nn.LSTM(
            input_size = INPUT_SIZE_RNN, 
            hidden_size = HIDDEN_SIZE / 2,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True
            )      

        ####  CNN 
        self.out3 = nn.Linear(HIDDEN_SIZE,2)
        self.conv1 = nn.Conv1d(in_channels = INPUT_SIZE, out_channels = OUT_CHANNEL, kernel_size = KERNEL_SIZE, stride = STRIDE)   
        self.maxpool = nn.MaxPool1d(kernel_size = MAXPOOL_NUM)
        ####  CNN 
        self.decoder_rnn = nn.LSTM(
            input_size = HIDDEN_SIZE, 
            hidden_size = Num,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=False
            )

    def forward_rnn(self, X_batch, X_len):  ### LSTM
    	batch_size = X_batch.shape[0]
    	dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        ###### Option I: final state
        #X_out, _ = self.rnn1(pack_X_batch, None) 
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]
        ####### Option II: hidden state
        _,(X_out,_) = self.encoder_rnn(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)
        X_out2 = X_out2[dd]    ## batch_size, HIDDEN_SIZE
    	return X_out2

    def forward_RCNN(self, X_batch, X_len):  ### CNN 
    	X_batch = torch.from_numpy(X_batch).float()
    	X_batch = Variable(X_batch)
    	batch_size = X_batch.shape[0]
        ### cnn + rnn
        X_batch_2 = X_batch.permute(0,2,1)  ## batch size,MAX_LENGTH,INPUT_SIZE => batch size,INPUT_SIZE,MAX_LENGTH
        X_batch_3 = self.conv1(X_batch_2)
        X_batch_4 = self.maxpool(X_batch_3)
        f_map = lambda x: max(int((int((x - KERNEL_SIZE) / STRIDE) + 1) / MAXPOOL_NUM),1)
        X_len2 = map(f_map, X_len)
        X_batch_4 = X_batch_4.permute(0,2,1)  ## => batch_size,MAX_LENGTH,INPUT_SIZE 

        X_out2 = self.forward_rnn(X_batch_4, X_len2)
        return X_out2  ### batch_size, HIDDEN_SIZE 


    def forward(self, X_batch, X_len):
        batch_size = X_batch.shape[0]

        X_out2 = self.forward_RCNN(X_batch, X_len)  ### batch_size, HIDDEN_SIZE 
        X_out2 = X_out2.view(batch_size, HIDDEN_SIZE, 1)
        X_out2 = X_out2.expand(batch_size, HIDDEN_SIZE, MAX_LENGTH)  ### => batch_size, HIDDEN_SIZE, MAX_LENGTH

        dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        #print(X_len_sort)
        X_batch_v = X_out2[dd1]
        X_batch_v = X_batch_v.permute(0,2,1)   ### => batch_size, HIDDEN_SIZE, MAX_LENGTH
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)
        X_out3, _ = self.decoder_rnn(pack_X_batch, None)
        X_out4 = torch.nn.utils.rnn.pad_packed_sequence(X_out3)
        #print(X_out4[1]) 
        data = X_out4[0]
        data = data.permute(1,0,2)
        data = data[dd]
        #print(data)
        data = F.log_softmax(data,2)
        X_out5 = data[0,:X_len[0],:]
        for i in range(1,batch_size):
            X_out5 = torch.cat([X_out5, data[i,:X_len[i],:]], 0)
        return X_out5 


LR = 1e-1 ### 4e-3 is ok,
EPOCH = 9

nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True)
print('Build network')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()


N = len(data_dict)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
def add(a,b):
    return a + b 

def cut_length(a):
    return a[-MAX_LENGTH:]

lamb = 4e-3
for epoch in range(EPOCH):
    loss_average = 0
    t1 = time()
    for i in range(iter_in_epoch):
        ##### train
        #print('Epoch ' + str(epoch)+ ': iter ' + str(i) , end = ' ')
        t11 = time()
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x , batch_len = data2array(data_dict[stt:endn])

        #### label 
        tmp = data_dict[stt:endn]
        tmp = map(cut_length, tmp)
        batch_label = reduce(add, tmp)
        batch_label = np.array(batch_label)
        batch_label = torch.from_numpy(batch_label)
        batch_label = Variable(batch_label)
        #print(batch_label.shape)
        
        ###  batch 

        output = nnet(batch_x, batch_len)
        #print(output.shape)
        loss = loss_crossentropy(output, batch_label)
        opt_.zero_grad()
        loss.backward()  ##retain_variables = True
        opt_.step()
        loss_value = loss.data[0]
        loss_average += loss_value
        t22 = time()
        print(str(t22 - t11) + ' seconds')

    t2 = time()
    print('Epoch ' + str(epoch) + ' costs ' + str(t2 - t1) + 'seconds ')
    print('Epoch '+str(epoch) + ' takes ' + str(int(t2-t1)) + ' sec. loss: ' + str(loss_average)[:6])



