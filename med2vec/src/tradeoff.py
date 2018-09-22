import matplotlib 
import numpy as np
import matplotlib.pyplot as plt


## pure rule learning
rule_num = [3, 7, 12, 32, 45, 54]
train_auc = [0.5113, 0.5197, 0.5262, 0.5433, 0.5480, 0.5492]
test_auc = [0.5064, 0.5169, 0.5231, 0.5337, 0.5380, 0.53899]

## prototype learning
rule_num2 = [1, 2, 3, 10, 20, 30, 40, 54]
train_auc2 = [0.505, 0.6655, 0.6687, 0.6702, 0.6727, 0.6743, 0.6759, 0.6762]
test_auc2 = [0.503 , 0.6633, 0.6662, 0.6677, 0.6697, 0.6713, 0.6722, 0.6722]

plt.plot(rule_num, train_auc, 'r-', label = 'Training AUC (Rule Learning)')
plt.plot(rule_num, test_auc, 'r--', label = 'Test AUC (Rule Learning)')

plt.plot(rule_num2, train_auc2, 'b:', label = 'Training AUC (PEARL)')
plt.plot(rule_num2, test_auc2, 'b-.', label = 'Test AUC (PEARL)')

plt.xlabel('Number of Rules', fontsize = 16)
plt.ylabel('Accuracy', fontsize = 16)
plt.legend()

#plt.show()
plt.savefig('./results/tradeoff.png')





