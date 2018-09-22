#!/bin/bash

########################################################
#####       data => rule, normalize
########################################################

### input is: ./data/test_data_1.txt & ./data/training_data_1.txt
### INPUT_FILE is changeable. 

TEST_FILE=./data/test_data_1.txt ###
cp ./data/training_data_1.txt ./data/train_data  ##


INPUT_FILE=./data/train_data
sed '1d' $TEST_FILE | awk -F "\t" '{print $3}' > ./data/test_feature
sed '1d' $TEST_FILE | awk '{print $1}' | sed 's/True/1/;s/False/0/' > ./data/test_label




#### test step 
wc -l ./data/training_data_1.txt ./data/train_data
####


	
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
./corels/corels -r 0.0000000015 -c 3 -p 1  data/training_selected_feature data/label > results/corels_rule_list




##############################
python2 ./src/deal_multihot.py --train_file ./data/train_data --multihot_train_data ./data/multihot_train_data  
python2 ./src/deal_multihot.py --train_file ./data/test_data_1.txt --multihot_train_data ./data/multihot_test_data   
### multihot
##############################

python2 ./src/multihot_prototype.py --multihot_train_data ./data/multihot_train_data --rulefile results/rule_data_list  \
   --train_label ./data/training_label --test_data ./data/multihot_test_data --test_label ./data/test_label 


python2 ./src/multihot_basis.py --multihot_train_data ./data/multihot_train_data --rulefile results/rule_data_list  \
   --train_label ./data/training_label --test_data ./data/multihot_test_data --test_label ./data/test_label 


