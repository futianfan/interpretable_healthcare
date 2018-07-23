INTERPRETABLE DEEP LEARNING FOR HEALTHCARE


For rule list part, we directly use the code from:
https://github.com/nlarusstone/corels

For word2vec, we use the code (https://github.com/Adoni/word2vec_pytorch)

src/neural_net-v1.py:  first version rule-list + lstm + prototype + binary classify
src/neural_net-v2.py:  second version rule + bi-lstm + prototype + binary classify
src/neural_net-v3.py:  bi-lstm + full-connect + binary classify 

v4: final state => hidden state
v5: full-connect => conv1d
v6: embedding => one-hot  add dim of "time stamp", optimal AUC = 0.64
v7: modify based on "v2", it doesn't work now 
v8: based on v6:  one-hot+timestamp => embedding, optimal AUC = 0.61, converge very slowly.
v9: embedding+lstm+prototype+softmax, don't success
v10: based on v8, embedding+lstm => embedding+cnn+lstm.   AUC = 0.67+


./run.sh


