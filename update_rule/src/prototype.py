'''

python2 ./src/prototype.py results/rule_data_list ./data/train_lstm_output.npy ./data/training_label  ./data/test_lstm_output.npy ./data/test_label  ./results/similarity

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
from numpy import linalg as LA 
from sklearn.metrics import roc_auc_score
torch.manual_seed(1)    # reproducible
np.random.seed(1)
################################################################################################################
print('STEP 1. read-in data')
##### input:   data/tmp3, results/corels_rule_list
##### output:  dict{rule: data-sample}
# read-in data:   data/tmp3
train_data = np.load(sys.argv[2])
test_data = np.load(sys.argv[4])

with open(sys.argv[3], 'r') as f_label:
    label_line = f_label.readlines()
    train_label = [int(line) for line in label_line]
train_label = np.array(train_label)

with open(sys.argv[5], 'r') as f_label:
    label_line = f_label.readlines()
    test_label = [int(line) for line in label_line]
test_label = np.array(test_label)

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
with open(sys.argv[1], 'r') as f_rule:
    lines = f_rule.readlines()
    rule_dict = {}
    for indx, line in enumerate(lines):
        rule_dict[indx] = [int(i) for i in line.split()]


print('num of matching data for each rule')
#print(rule_dict[0])

print('NN training')
INPUT_SIZE = 100 
HIDDEN_SIZE = train_data.shape[1]
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 50
BATCH_SIZE = 256
PROTOTYPE_NUM = len(rule_dict)
NUM_HIGH_WAY = 2


class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   

        #self.out1 = nn.Linear(PROTOTYPE_NUM, OUT_SIZE)
        #self.out2 = nn.Linear(OUT_SIZE, 2)
        self.out3 = nn.Linear(PROTOTYPE_NUM, 2)
    	## PROTOTYPE_NUM, HIDDEN_SIZE 
    	self.prototype = torch.zeros(PROTOTYPE_NUM, HIDDEN_SIZE)
    	self.prototype = Variable(self.prototype, requires_grad = False)

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

    def forward(self, X_batch):   
        batch_size = X_batch.shape[0]
        #X_batch = torch.from_numpy(X_batch) 
        #X_batch = Variable(X_batch)
        X_out2 = self.forward_highway(X_batch)

        X_out2 = X_out2.view(batch_size,HIDDEN_SIZE,1)
        X_out3 = X_out2.expand(batch_size, HIDDEN_SIZE, PROTOTYPE_NUM)
        prtt = self.prototype.view(1, HIDDEN_SIZE, PROTOTYPE_NUM)
        prtt = prtt.expand(batch_size, HIDDEN_SIZE, PROTOTYPE_NUM)
        X_diff = (X_out3 - prtt)**2
        X_out4 = torch.sum(X_diff, 1)  ### batch_size, PROTOTYPE_NUM
        ###### prototype
        #X_out5 = F.relu(self.out1(X_out4))
        #X_out6 = F.softmax(self.out2(X_out5))
        X_out6 = F.softmax(self.out3(X_out4))
        matching_loss, _ = torch.min(X_out4,1)
        matching_loss = torch.mean(matching_loss)
        return X_out6, matching_loss

    def generate_prototype(self, data_dict, rule_dict):
        p = Variable(torch.zeros(PROTOTYPE_NUM, HIDDEN_SIZE))
        for i in range(len(rule_dict)):
            #data = [data_dict[j] for j in rule_dict[i]]
            data = data_dict[rule_dict[i]]
            leng = len(data)
            T = np.ceil(leng * 1.0 / BATCH_SIZE)
            T = int(T)
            for j in range(T):
                bgn = j * BATCH_SIZE
                endn = min(leng, bgn + BATCH_SIZE)
                if bgn >= endn:
                    continue
                batch_data = data[bgn:endn]
                batch_data = Variable(torch.from_numpy(batch_data))
                batch_X_out = self.forward_highway(batch_data)
                if j==0:
                    X_out = batch_X_out
                else:
                    X_out = torch.cat([X_out, batch_X_out], 0)
            #self.prototype[i,:] = torch.mean(X_out, 0)
            p[i,:] = torch.mean(X_out, 0)
        self.prototype = p.data 
        self.prototype = Variable(self.prototype, requires_grad = False)
        #self.prototype.requires_grad = False 
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
        batch_x = data_dict[stt:endn]
        if batch_x.shape[0] == 0:
            break
        batch_x = Variable(torch.from_numpy(batch_x))
        output,_ = nnet(batch_x)
        output_data = output.data 
        for j in range(output_data.shape[0]):
            #print(str(data_label[stt + j]) + ' ' + str(output_data[j][0]))
            #fout.write(str(data_label[stt + j]) + ' ' + str(output_data[j][0]) + '\n')
            y_pred.append(output_data[j][1])
            y_label.append(data_label[stt+j])
    auc = roc_auc_score(y_label, y_pred)
    print('AUC of Epoch ' + str(epoch) + ' is ' + str(auc)[:5])

def similarity(nnet, train_data, rule_dict, fout):
    nnet.generate_prototype(train_data, rule_dict)
    prototype_np = nnet.prototype.data.numpy()
    prototype_norm = LA.norm(prototype_np, axis = 1).reshape(-1,1) * np.ones((1,HIDDEN_SIZE))
    prototype_np = prototype_np / prototype_norm
    similarity_for_prototype = np.zeros((PROTOTYPE_NUM, 1))
    N = len(train_data)
    iter_num = int(np.ceil(N * 1.0 / BATCH_SIZE))
    for i in range(iter_num):
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x = train_data[stt:endn]
        if batch_x.shape[0] == 0:
            break
        batch_x = Variable(torch.from_numpy(batch_x))
        X_out2 = nnet.forward_highway(batch_x)
        X = X_out2.data.numpy()
        X_norm = (LA.norm(X, axis = 1)).reshape(-1, 1) * np.ones((1, HIDDEN_SIZE))
        X = X / X_norm  ### batch_size, HIDDEN_SIZE 
        similarity_for_prototype += (np.sum(np.array(np.mat(prototype_np) * np.mat(X.T)), 1)).reshape(PROTOTYPE_NUM, 1)
    f = open(fout, 'w')
    for i in range(PROTOTYPE_NUM):
        f.write(str(similarity_for_prototype[i,0]) + ' ')
    f.close()



    ## write the results 




LR = 1e-1 ### 4e-3 is ok,  
nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, PROTOTYPE_NUM, OUT_SIZE,  BATCH_FIRST = True)
print('Build network')
nnet.generate_prototype(train_data, rule_dict)
print('Generate prototype')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

N = len(train_data)
iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
EPOCH = 20
lamb = 0
for epoch in range(EPOCH):
    if epoch >= 0 and epoch % 1 == 0:
        t1 = time()
        #print('test: ', end = ' ')
        #test_X(nnet, data_dict, label, epoch)
        test_X(nnet, test_data, test_label, epoch)
        t2 = time()
        print('test cost ' + str(int(t2-t1)) + ' sec. ')
    loss_average = 0
    t1 = time()
    for i in range(iter_in_epoch):
        ##### train
        #print('Epoch ' + str(epoch)+ ': iter ' + str(i) , end = ' ')
        tt1 = time()
        nnet.generate_prototype(train_data, rule_dict)
        tt2 = time()
        #print('generating prototype cost ' + str(tt2 - tt1)[:6] + ' sec')
        t11 = time()
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        if stt == endn:
            continue 
        batch_x = train_data[stt:endn]
        batch_label = train_label[stt:endn]
        batch_label = torch.from_numpy(batch_label)
        batch_label = Variable(batch_label)
        batch_x = Variable(torch.from_numpy(batch_x))
        output, matching_loss = nnet(batch_x)
        loss = loss_crossentropy(output, batch_label) # + lamb * matching_loss
        opt_.zero_grad()
        loss.backward()  ## retain_variables = True ,   retain_graph = True
        #loss.backward(retain_graph = True)
        #print(nnet.prototype.requires_grad)
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

similarity(nnet, train_data, rule_dict, sys.argv[6])

'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''






