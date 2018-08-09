INTERPRETABLE DEEP LEARNING FOR HEALTHCARE


For rule list part, we directly use the code from:
https://github.com/nlarusstone/corels

For word2vec, we use the code (https://github.com/Adoni/word2vec_pytorch)

src/neural_net-v1.py:  first version rule-list + lstm + prototype + binary classify

src/neural_net-v2.py:  second version rule + bi-lstm + prototype + binary classify

src/neural_net-v3.py:  bi-lstm + full-connect + binary classify 

v4: final state => hidden state

v5: full-connect => conv1d

v6: embedding => one-hot  add dim of "time stamp", optimal AUC = 0.64,   
	52.3seconds/epoch, totally epoch


v7: modify based on "v2", it doesn't work now 

v8: based on v6:  one-hot+timestamp => embedding, optimal AUC = 0.61, converge very slowly.

v9: embedding+lstm+prototype+softmax, don't success

v10: based on v8, embedding+lstm => embedding+cnn+lstm.  
 AUC = 0.672+ => 0.684+ (lr=1e-1,version 13)   20sec/epoch  

v11: based on v10, embedding+cnn+lstm+full-connect => embedding+cnn+lstm+highway+full-connect.   
AUC = 0.671+

v12: based on v11, embedding+cnn+lstm+highway+full-connect => embedding+cnn+lstm+highway+prototype+full-connect, doesn't work.

v13: see v10. same with v10

v14: based on v10/13, embedding + cnn + lstm + full-connect, then save the output of LSTM in ./data/

v15: use the output of v14, prototype. AUC=0.676 (highway=1:0.676,2:0.676,3:0.678+, 4:0.636+).   don't BP grad for prototype.

v14 + v15:

v16: same with v15, but do BP gradient for prototype. it's very slow, AUC=0.672.

v17: it is based on v15, add some test procedure
v18: encoder-decoder seq2seq 



./run.sh


