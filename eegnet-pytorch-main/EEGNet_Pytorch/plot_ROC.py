import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import os

result_path = "/root/autodl-tmp/EEGNet/EEGNet_Pytorch/eegnet_result"
n_splits = 10  # 假设是10折交叉验证

# 存储所有插值后的 TPR，用于计算平均值
all_tprs = []
mean_fpr = np.linspace(0, 1, 100)

aucs = []

for i in range(n_splits):
    fpr = np.load(os.path.join(result_path, f"fpr_fold{i+1}.npy"))
    tpr = np.load(os.path.join(result_path, f"tpr_fold{i+1}.npy"))

    # 插值到统一的 mean_fpr 上
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    all_tprs.append(interp_tpr)

    fold_auc = auc(fpr, tpr)
    aucs.append(fold_auc)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f"Fold {i + 1} (AUC = {fold_auc:.2f})")

# 计算平均 TPR 和标准差
mean_tpr = np.mean(all_tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=r"Mean ROC (AUC = %.2f $\pm$ %.2f)" % (mean_auc, std_auc),
         lw=2, alpha=0.8)

# 阴影区域表示标准差
std_tpr = np.std(all_tprs, axis=0)
tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                 label=r'$\pm$ 1 std. dev.')

# 设置图像参数
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Mean ROC Curve Across 10 Folds")
plt.legend(loc="lower right")
plt.grid(True)

# 保存图像
save_path = os.path.join(result_path, "mean_roc_curve.png")
plt.savefig(save_path, dpi=300)
plt.show()
