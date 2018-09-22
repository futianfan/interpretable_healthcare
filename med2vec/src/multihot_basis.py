'''

python2 ./src/multihot_prototype.py results/rule_data_list ./data/train_lstm_output.npy ./data/training_label  ./data/test_lstm_output.npy ./data/test_label  ./results/similarity
python2 ./src/multihot_prototype.py --multihot_train_data ./data/multihot_train_data --rulefile results/rule_data_list \
    --train_label ./data/training_label --test_data ./data/multihot_test_data --test_label ./data/test_label 
'''

from __future__ import print_function
import argparse
import sys
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np 
from time import time
from numpy import linalg as LA 
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument("--rulefile", help=" rulefile ", type=str)
parser.add_argument("--multihot_train_data", help="multihot train file ", type=str)
parser.add_argument("--train_label", help=" train label ", type=str)
parser.add_argument("--test_data", help="test data", type=str)
parser.add_argument("--test_label", help="test label", type=str)
args = parser.parse_args()


################################################################################################################
print('STEP 1. read-in data')

with open(args.multihot_train_data, 'r') as f_train_data:
    train_data_lines = f_train_data.readlines()

with open(args.test_data, 'r') as f_test_data:
    test_data_lines = f_test_data.readlines()

with open(args.train_label, 'r') as f_label:
    label_line = f_label.readlines()
    train_label = [int(line) for line in label_line]
train_label = np.array(train_label)

with open(args.test_label, 'r') as f_label:
    label_line = f_label.readlines()
    test_label = [int(line) for line in label_line]
test_label = np.array(test_label)

Dim = 1869

################################################################################################################
################################################################################################################ 
print('STEP 2. read-in rule, assign data to rule')
# read-in rule:   results/corels_rule_list
f4 = lambda x: True if 'if ' in x and ' then ' in x else False
def rule2num(string):
    bgn = string.find('_') + 1
    endn = string.find('=')
    fea_num = int(string[bgn:endn])
    lab = 1 if 'yes' else 0
    return fea_num,lab

## assign data to rule
with open(args.rulefile, 'r') as f_rule:
    lines = f_rule.readlines()
    rule_dict = {}
    for indx, line in enumerate(lines):
        rule_dict[indx] = [int(i) for i in line.split()]


print('NN training')
INPUT_SIZE = 100 
HIDDEN_SIZE = 50  ### RNN's output: hidden size
OUT_SIZE = 30 
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 5
BATCH_SIZE = 16
PROTOTYPE_NUM = len(rule_dict)
NUM_HIGH_WAY = 2
MED2VEC_SIZE = 30

'''
import numpy as np
Dim = 3
MAX_LENGTH = 5
'''
def lst2vec(lst):
    vec = np.zeros((1,Dim) ,dtype = float)
    if len(lst) == 0:
        return vec 
    lst = [int(i) for i in lst.split()]
    for i in lst:
        vec[0,i] = 1
    return vec 

def line2matrix(line):
    line = line.rstrip()
    line = line.split('\t')[:MAX_LENGTH]
    line = [[]] * (MAX_LENGTH - len(line)) + line 
    mat = map(lst2vec, line)
    mat2 = mat[0]
    for i in range(1,len(mat)):
        mat2 = np.concatenate((mat2, mat[i]),0)
    return mat ## MAX_LENGTH, Dim


class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE, Dim, MED2VEC_SIZE, BATCH_FIRST = True):
        super(RLP, self).__init__()
        self.Wc = nn.Linear(Dim, MED2VEC_SIZE)
        self.Vc = nn.Linear(MED2VEC_SIZE, MED2VEC_SIZE)
        self.rnn1 = nn.LSTM(
            input_size = MED2VEC_SIZE, 
            hidden_size = HIDDEN_SIZE/2,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True)
        self.out1 = nn.Linear(HIDDEN_SIZE, 2)
        self.f = F.relu
        ##########################################################
        self.num_layers = NUM_HIGH_WAY
        self.nonlinear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.linear = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])
        self.gate = nn.ModuleList([nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE) for _ in range(self.num_layers)])


    def forward_encoderNN(self, X_line):
        batch_size = len(X_line)
        Xdata = map(line2matrix, X_line)
        X_tensor = np.expand_dims(Xdata[0], axis = 0)
        for i in range(1, batch_size):
            tt = np.expand_dims(Xdata[i], axis = 0)
            X_tensor = np.concatenate((X_tensor, tt), axis = 0)
        X_batch = torch.from_numpy(X_tensor).float()  ### batch_size, MAX_LENGTH, Dim
        X_batch = Variable(X_batch)
        X_batch = X_batch.view(-1, Dim)
        X_med2vec = self.f(self.Wc(X_batch))  ### batch_size * MAX_LENGTH, MED2VEC_SIZE 
        ##X_med2vec = self.f(self.Vc(X_med2vec))  ## MED2VEC_SIZE
        X_med2vec = X_med2vec.view(batch_size, MAX_LENGTH, MED2VEC_SIZE) 
        X_len = [MAX_LENGTH for i in range(batch_size)]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_med2vec, X_len, batch_first=True)
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)
        return X_out2

    def forward_highway(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

    def forward(self, X_line): 
        X_out2 = self.forward_encoderNN(X_line)
        X_out2 = self.forward_highway(X_out2)
        X_out3 = F.softmax(self.out1(X_out2))
        return X_out3, 0  



def test_X(nnet, test_data_lines, test_label, epoch):
    ## data_dict, data_label is a list, [0,1,1,1,0 0 0 0 ]
    N_test = len(test_data_lines)
    #fout = open('./results/test_result_of_epoch_' + str(epoch), 'w')
    iter_num = int(np.ceil(N_test * 1.0 / BATCH_SIZE))
    y_pred = []
    y_label = []
    for i in range(iter_num):
        stt = i * BATCH_SIZE
        endn = min(N_test, stt + BATCH_SIZE)
        batch_x = test_data_lines[stt:endn]
        if len(batch_x) == 0:
            break
        output,_ = nnet(batch_x)
        output_data = output.data 
        for j in range(output_data.shape[0]):
            y_pred.append(output_data[j][1])
            y_label.append(test_label[stt+j])
    auc = roc_auc_score(y_label, y_pred)
    print('AUC of Epoch ' + str(epoch) + ' is ' + str(auc)[:5])





LR = 1e-0 ### 4e-3 is ok,  
nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE, Dim, MED2VEC_SIZE, BATCH_FIRST = True)
print('Build network')
#nnet.generate_prototype(train_data, rule_dict)
#print('Generate prototype')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

N = len(train_data_lines)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
EPOCH = 30
lamb = 0
for epoch in range(EPOCH):
    '''
    if epoch >= 0 and epoch % 1 == 0:
        t1 = time()
        #print('test: ', end = ' ')
        #test_X(nnet, data_dict, label, epoch)
        test_X(nnet, test_data, test_label, epoch)
        t2 = time()
        print('test cost ' + str(int(t2-t1)) + ' sec. ')
    loss_average = 0
    '''
    test_X(nnet, test_data_lines, test_label, epoch)
    loss_average = 0 
    t1 = time()
    for i in range(iter_in_epoch):
        t11 = time()
        ## print('1')
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        if stt == endn:
            continue 
        batch_x = train_data_lines[stt:endn]
        batch_label = train_label[stt:endn]
        batch_label = Variable(torch.from_numpy(batch_label))
        output, _ = nnet(batch_x)
        loss = loss_crossentropy(output, batch_label) # + lamb * matching_loss
        opt_.zero_grad()
        loss.backward()  ## retain_variables = True ,   retain_graph = True
        #loss.backward(retain_graph = True)
        opt_.step()    
        #opt_.zero_grad()
        loss_value = loss.data[0]
        loss_average += loss_value
        ##loss.backward(retain_variables = False) ##########  
        #print('loss is ' + str(loss_value)[:7], end = ' ')
        t22 = time()
        #print('training cost ' + str((t22 - t11))[:5] + ' seconds')
        #print('Epoch: ' + str(epoch) + ", " + str(i) + "/"+ str(iter_in_epoch)+ ': loss value is ' + str(loss.data[0]))
        ##### train
    l_his.append(loss_average)
    t2 = time()
    print('Epoch '+str(epoch) + ' takes ' + str((t2-t1)) + ' sec. loss: ' + str(loss_average)[:6] )


'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''






