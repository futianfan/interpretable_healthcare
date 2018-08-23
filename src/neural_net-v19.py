'''
python ./src/neural_net.py data/training_model_by_word2vec_1.vector data/tmp3 ./data/snow.Y ./data/test_data_1_3.txt ./data/test_snow.Y
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
from sklearn.cluster import KMeans
torch.manual_seed(1)    # reproducible
np.random.seed(1)
################################################################################################################ 
print('read-in word2vec')
f1 = lambda x:x.split()[0]
f2 = lambda x:np.array([float(i) for i in x.split()[1:]])
with open(sys.argv[1], 'r') as word2vec_file:
	lines = word2vec_file.readlines()
	lines = lines[1:]
	id2vec = {f1(line):f2(line) for line in lines}
################################################################################################################
print('read-in training data')
##### input:   data/tmp3, results/corels_rule_list
##### output:  dict{rule: data-sample}
# read-in data:   data/tmp3
f3 = lambda x:[int(i) for i in x.rstrip().split()]
with open(sys.argv[2], 'r') as raw_data:
	data_lines = raw_data.readlines()
	#data_dict = {i:f3(line) for i,line in enumerate(data_lines)}
	data_dict = [f3(line) for i,line in enumerate(data_lines)]
################################################################################################################
print('read-in training label')
with open(sys.argv[3], 'r') as f_label:
    label_line = f_label.readlines()
    label = []
    for line in label_line:
        label.append(int(line))
label = np.array(label)
################################################################################################################
print('read-in test data')
with open(sys.argv[4], 'r') as f_test:
    lines = f_test.readlines()
    test_data_dict = [f3(line) for i,line in enumerate(lines)]
################################################################################################################
print('read-in test label')
with open(sys.argv[5], 'r') as f_label:
    label_line = f_label.readlines()
    test_label = []
    for line in label_line:
        test_label.append(int(line))
test_label = np.array(test_label)
################################################################################################################
print('basic config')
INPUT_SIZE = 100 ## embedding size
HIDDEN_SIZE = 50 
OUT_SIZE = 30
NUM_LAYER = 1
BATCH_FIRST = True 
MAX_LENGTH = 60
BATCH_SIZE = 256 
KERNEL_SIZE = 10
STRIDE = 1
CNN_OUT_SIZE = int((MAX_LENGTH - KERNEL_SIZE)/STRIDE) + 1
OUT_CHANNEL = INPUT_SIZE
MAXPOOL_NUM = 3
INPUT_SIZE_RNN = OUT_CHANNEL
Cluster_num = 30
interpret_OUT_SIZE = 30


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
################################################################################################################
print('build NN')
class RLP(torch.nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUT_SIZE, interpret_OUT_SIZE,  BATCH_FIRST = True):
        super(RLP, self).__init__()   
        self.rnn1 = nn.LSTM(
            input_size = INPUT_SIZE_RNN, 
            hidden_size = HIDDEN_SIZE / 2,
            num_layers = NUM_LAYER,
            batch_first = BATCH_FIRST,
            bidirectional=True
            )      
        ## Option I
        self.out1 = nn.Linear(HIDDEN_SIZE, OUT_SIZE)
        self.out2 = nn.Linear(OUT_SIZE, 2)
        ## Option II
        self.out3 = nn.Linear(HIDDEN_SIZE,2)
        ## CNN
        self.conv1 = nn.Conv1d(in_channels = INPUT_SIZE, out_channels = OUT_CHANNEL, kernel_size = KERNEL_SIZE, stride = STRIDE)   
        self.maxpool = nn.MaxPool1d(kernel_size = MAXPOOL_NUM)
        ## interpretable
        self.interpret_out = nn.Linear(HIDDEN_SIZE, interpret_OUT_SIZE)

    def forward_rnn(self, X_batch, X_len):
    	batch_size = X_batch.shape[0]
    	dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)

        ### Option I: final state
        #X_out, _ = self.rnn1(pack_X_batch, None) 
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]

        ### Option II: hidden state 
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)

        X_out2 = X_out2[dd] ## re-order   ## batch_size, HIDDEN_SIZE
    	return X_out2

    def forward_rcnn(self, X_batch, X_len):
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
        X_out2 = self.forward_rnn(X_batch_4, X_len2)
    	#X_out2 = self.forward_rnn(X_batch, X_len)
        return X_out2 

    def forward(self, X_batch, X_len):
        X_out2 = self.forward_rcnn(X_batch, X_len)
        ### full connected
        X_out6 = F.softmax(self.out3(X_out2))
        return X_out6

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

###########################################################################################
N = len(data_dict)
## Cluster_num = 30
data2cluster = np.random.randint(0,Cluster_num, N)  ## Initialization  


def data_index_for_each_cluster(data2cluster, Cluster_num, N):
    clusters_data = [[] for i in range(Cluster_num)]
    for i in range(Cluster_num):
        ff = lambda x: True if i == data2cluster[x] else False 
        clusters_data[i] = filter(ff, [j for j in range(N)])
    return clusters_data

###########################################################################################




LR = 2e-2 ### 4e-3 is ok,  
nnet  = RLP(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYER, OUT_SIZE, interpret_OUT_SIZE, BATCH_FIRST = True)
print('Build network')
opt_  = torch.optim.SGD(nnet.parameters(), lr=LR)  # SGD    Adam 
loss_crossentropy = torch.nn.CrossEntropyLoss()
l_his = []

iter_in_epoch = int(np.ceil(N * 1.0 /BATCH_SIZE))
#test_N = test_query.shape[0]
#test_iter_in_epoch = int(test_N / batch_size)
EPOCH = 450
lamb = 4e-3
for epoch in range(EPOCH):
    if epoch >= 0 and epoch % 3 == 0:
        t1 = time()
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
    ####### cluster 
    ### generate code for each data sample
    for i in range(iter_in_epoch):
        stt = i * BATCH_SIZE
        endn = min(N, stt + BATCH_SIZE)
        batch_x, batch_len = data2array(data_dict[stt:endn])
        hidden_state = nnet.forward_rcnn(batch_x, batch_len)
        if i == 0:
            X_code = hidden_state
        else:
            X_code = torch.cat([X_code, hidden_state], 0)    
    ### re-cluster

    kmeans = KMeans(n_clusters=Cluster_num, random_state=0).fit(X_code)  ##  http://scikit-learn.org/stable/modules/clustering.html
    labels = list(kmeans.labels_) 
    ### get centroid and target value for each cluster 
    ### 
    cluster2data = data_index_for_each_cluster(labels)
    centroid_matrix = Variable(torch.zeros(Cluster_num, HIDDEN_SIZE))
    target_value = Variable(torch.zeros(Cluster_num, interpret_OUT_SIZE))
    for i in range(Cluster_num):
        centroid_matrix[i,:] = torch.mean(X_code[cluster2data[i]], 0).view(1,-1)
        










    l_his.append(loss_average)
    t2 = time()
    print('Epoch '+str(epoch) + ' takes ' + str(int(t2-t1)) + ' sec. loss: ' + str(loss_average)[:6])


'''
plt.plot(l_his, label='SGD')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.savefig('sgd.jpg')
plt.show()
'''






