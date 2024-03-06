from matplotlib import pyplot as plt

train = plt.plot(avg_loss_1, color='blue', label='training set')
test = plt.plot(avg_loss_2, color='red', label='test set')
plt.legend()   
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xscale('log')
plt.xlabel('fpr_log')
plt.ylabel('tpr')

plt.figure()
plt.plot(best_fpr,best_tpr)
plt.xscale('log')
plt.xlim(10**-3,10**0)
plt.xlabel('partial_fpr_log')
plt.ylabel('tpr')

#plt.figure()
#plt.plot(best_roc_log_fpr,best_roc_tpr)
#plt.xlabel('partial_fpr_log_2')
#plt.ylabel('tpr')

#p_auc = auc(best_roc_log_fpr,best_roc_tpr)