#!/bin/bash

INPUT_FILE=./data/training_data_1.txt
sed '1d' $INPUT_FILE | awk '{print $1}' | sed 's/True/1/;s/False/0/' > ./data/tmp1
cp ./data/tmp1 ./data/snow.Y
sed '1d' $INPUT_FILE | awk '{print $1}' | sed 's/True/0/;s/False/1/' > ./data/tmp2
sed '1d' $INPUT_FILE | awk -F "\t" '{print $3}' > ./data/tmp3   ### tmp 
n=`python src/findmax_N.py ./data/tmp3`
((n=n+1))

sed '1d' data/test_data_1.txt | awk -F "\t" '{print $3}' > ./data/test_data_1_3.txt
sed '1d' data/test_data_1.txt | awk '{print $1}' | sed 's/True/1/;s/False/0/' > ./data/tmp1
cp ./data/tmp1 ./data/test_snow.Y

cd data
s1=`tr '\n' ' ' < tmp1`
s2=`tr '\n' ' ' < tmp2`
echo -e '{label:Yes} \c' > label
echo $s1 >> label
echo -e '{label:No} \c'>>label
echo $s2 >> label
cd -

#python ./src/generate_X_data.py $n ./data/tmp3 ./data/snow.X

python ./src/generate_X_using_feature_selection.py $n ./data/snow.Y ./data/tmp3 ./data/snow.X_new
./corels/corels -r 0.0000000015 -c 3 -p 1  data/snow.X_new data/label > results/corels_rule_list
######################## generate corels rule list ########################

###################################################
########################  NN
## 1. word embedding    
# input is data/tmp3 
python ./word2vec_tool/word2vec.py ./data/tmp3 ./data/id2vec.txt
## 2. LSTM + 3. rule -> prttype
  ## rule r -> x -> prttype

###   python ./src/neural_net.py data/id2vec.txt data/tmp3 results/corels_rule_list ./data/snow.Y
((n=n+1))
## python ./src/neural_net.py data/id2vec.txt data/training_data_1.txt results/corels_rule_list ./data/snow.Y $n
## python ./src/neural_net.py data/training_data_1.txt ./data/snow.Y $n
## python ./src/neural_net.py data/training_data_1.txt ./data/snow.Y $n ./data/test_data_1.txt
#python ./src/neural_net.py data/training_model_by_word2vec_1.vector data/tmp3 results/corels_rule_list ./data/snow.Y ./data/test_data_1_3.txt
python ./src/neural_net.py data/training_model_by_word2vec_1.vector data/tmp3 results/corels_rule_list ./data/snow.Y ./data/test_data_1_3.txt ./data/test_snow.Y

### !!find key disease!!
python ./src/find_key_factor.py data/tmp3 50 data/key_factor 


cat results/test_result_of_epoch_5 | awk '{ sum += $2; } END { print "average = " sum/NR }'

grep ^0 results/test_result_of_epoch_36 | awk '{ sum += $2; } END { print "average = " sum/NR}'
grep ^1 results/test_result_of_epoch_36 | awk '{ sum += $2; } END { print "average = " sum/NR}'


