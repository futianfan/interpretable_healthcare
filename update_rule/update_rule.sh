#!/bin/bash

########################################################
#####       data => rule, normalize
########################################################

### input is: ./data/test_data_1.txt & ./data/training_data_1.txt
### INPUT_FILE is changeable. 

TEST_FILE=./data/test_data_1.txt
cp ./data/training_data_1.txt ./data/train_data
INPUT_FILE=./data/train_data
sed '1d' $TEST_FILE | awk -F "\t" '{print $3}' > ./data/test_feature
sed '1d' $TEST_FILE | awk '{print $1}' | sed 's/True/1/;s/False/0/' > ./data/test_label




####
wc -l ./data/training_data_1.txt ./data/train_data
####

############################################################################
## FOR iteration
END=5
for((i=1;i<=END;i++)); do
sed '1d' $INPUT_FILE | awk '{print $1}' | sed 's/True/1/;s/False/0/' > ./data/training_label
sed '1d' $INPUT_FILE | awk '{print $1}' | sed 's/True/0/;s/False/1/' > ./data/tmp2
sed '1d' $INPUT_FILE | awk -F "\t" '{print $3}' > ./data/training_feature   ### tmp 
n=`python src/findmax_N.py ./data/training_feature`
((n=n+1))


cd data
s1=`tr '\n' ' ' < training_label`
s2=`tr '\n' ' ' < tmp2`
echo -e '{label:Yes} \c' > label
echo $s1 >> label
echo -e '{label:No} \c'>>label
echo $s2 >> label
cd -

#### feature selection   1-2 minutes 
python2 ./src/generate_X_using_feature_selection.py $n ./data/training_label ./data/training_feature ./data/training_selected_feature
#### rule learning
./corels/corels -r 0.0000000015 -c 3 -p 1  data/training_selected_feature data/label > results/corels_rule_list


##### output is: 

########################################################
#####       NN + prototype training
########################################################
### 18 seconds per EPOCH   18*20/60 = 6 min 
python2 ./src/rcnn_fc_softmax.py data/training_model_by_word2vec_1.vector data/training_feature results/corels_rule_list ./data/training_label ./data/test_feature ./data/test_label ./data/train_lstm_output.npy ./data/test_lstm_output.npy ./results/rule_data_list



python2 ./src/prototype.py results/rule_data_list ./data/train_lstm_output.npy ./data/training_label  ./data/test_lstm_output.npy ./data/test_label ./results/similarity 

python2 ./src/add_data.py $INPUT_FILE results/rule_data_list results/similarity tmp 5
mv tmp $	
done
## END FOR








