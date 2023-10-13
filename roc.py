import pandas as pd
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import numpy as np


fpr = pd.read_csv("fpr.csv", index_col=0, dtype=np.float32).to_numpy()
tpr = pd.read_csv("tpr.csv", index_col=0, dtype=np.float32).to_numpy()
auc = auc(fpr, tpr)  # 计算平均AUC值

plt.plot(fpr, tpr, color='purple', label=r'm5UMCB  (AUC=%0.4f)' % auc, lw=2, alpha=.8)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()