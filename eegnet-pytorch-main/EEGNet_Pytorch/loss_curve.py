import numpy as np
import matplotlib.pyplot as plt
import os

# 路径
result_path = "E:\lin\stroke\stroke_GCN_ver2\eegnet-pytorch-main\EEGNet_Pytorch\eegnet_result"

# 加载数据
train_losses = np.load(os.path.join(result_path, "train_losses.npy"), allow_pickle=True)
valid_losses = np.load(os.path.join(result_path, "valid_losses.npy"), allow_pickle=True)  # 或 test_losses.npy

# 转为 numpy 数组，形状：(n_folds, n_epochs)
train_losses = np.array(train_losses)
valid_losses = np.array(valid_losses)

n_folds, n_epochs = train_losses.shape

# 画图
plt.figure(figsize=(10, 6))

# 每一折的曲线
for i in range(n_folds):
    plt.plot(train_losses[i], color='lightcoral', alpha=0.3)
    plt.plot(valid_losses[i], color='lightblue', alpha=0.3)

# 画平均曲线
plt.plot(np.mean(train_losses, axis=0), color='red', label='Train Loss (mean)', linewidth=2)
plt.plot(np.mean(valid_losses, axis=0), color='blue', label='Valid Loss (mean)', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Validation Loss Across Folds")
plt.legend()
plt.grid(True)

# 保存图像
plt.tight_layout()
# plt.savefig(os.path.join(result_path, "loss_curve.png"), dpi=300)
plt.show()
