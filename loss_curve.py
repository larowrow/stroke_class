# import numpy as np
# import matplotlib.pyplot as plt

# band = 'Gamma'
# connetc = "plv"

# lr = 0.0005
# lr_step = 70
# epochs = 150
# # 读取保存的训练和验证数据
# train_losses = np.load(f"data/time_8/{band}/1loss_curve055/{connetc}_{lr}_{lr_step}_{epochs}_train_losses.npy")  # shape: (10, epochs)
# train_accs = np.load(f"data/time_8/{band}/1loss_curve055/{connetc}_{lr}_{lr_step}_{epochs}_train_accs.npy")      # shape: (10, epochs)
# valid_losses = np.load(f"data/time_8/{band}/1loss_curve055/{connetc}_{lr}_{lr_step}_{epochs}_valid_losses.npy")  # shape: (10, epochs)
# valid_accs = np.load(f"data/time_8/{band}/1loss_curve055/{connetc}_{lr}_{lr_step}_{epochs}_valid_accs.npy")      # shape: (10, epochs)
# auc = np.load(f"data/time_8/{band}/1loss_curve055/{connetc}_{lr}_{lr_step}_{epochs}_aucs.npy")

# # 获取 epochs 的数量
# epochs = train_losses.shape[1]

# # 计算每个指标的均值
# mean_train_loss = np.mean(train_losses, axis=0)
# mean_train_acc = np.mean(train_accs, axis=0)
# mean_valid_loss = np.mean(valid_losses, axis=0)
# mean_valid_acc = np.mean(valid_accs, axis=0)
# mean_auc = np.mean(auc, axis=0)
# # 创建一个新的图形
# plt.figure(figsize=(12, 5))
# # 绘制训练和验证损失曲线
# plt.subplot(1, 3, 1)

# for i in range(10):  # 绘制每一折的损失曲线
#     plt.plot(range(epochs), train_losses[i], color='blue', alpha=0.5)  # 训练损失，蓝色
#     plt.plot(range(epochs), valid_losses[i], color='red', alpha=0.5)   # 验证损失，红色

# # 绘制均值损失曲线
# plt.plot(range(epochs), mean_train_loss, color='black', label='Mean Train Loss', linewidth=2)##00008B
# plt.plot(range(epochs), mean_valid_loss, color='black', label='Mean Valid Loss', linewidth=2)##8B0000

# plt.title('Loss Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc='upper right')

# # 绘制训练和验证准确率曲线
# plt.subplot(1, 3, 2)
# for i in range(10):  # 绘制每一折的准确率曲线
#     plt.plot(range(epochs), train_accs[i], color='green', alpha=0.5)  # 训练准确率，绿色
#     plt.plot(range(epochs), valid_accs[i], color='yellow', alpha=0.5) # 验证准确率，黄色

# # 绘制均值准确率曲线
# plt.plot(range(epochs), mean_train_acc, color='black', label='Mean Train Accuracy', linewidth=2)##006400
# plt.plot(range(epochs), mean_valid_acc, color='black', label='Mean Valid Accuracy', linewidth=2)#orange

# plt.title('Accuracy Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')

# # 绘制训练和验证损失曲线
# plt.subplot(1, 3, 3)
# for i in range(10):  # 绘制每一折的损失曲线
#     plt.plot(range(epochs), auc[i], color='#D8B9D6', alpha=0.5)  # 训练损失，蓝色

# # 绘制均值损失曲线
# plt.plot(range(epochs), mean_auc, color='black', label='Mean valid auc', linewidth=2)#4B0082

# plt.title('Auc Curve')
# plt.xlabel('Epoch')
# plt.ylabel('Auc')
# plt.legend(loc='upper right')


# # 显示图形
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# 配置参数
band = 'Gamma'
connetc = "plv"
lr = 0.005
lr_step = 50
epochs_expect = 150  # 预期的 epochs 数量
weight_decay=1e-4
dropoutp=0.4
# 加载数据
train_losses = np.load(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_losses.npy")
train_accs = np.load(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_accs.npy")
valid_losses = np.load(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_losses.npy")
valid_accs = np.load(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_accs.npy")
auc = np.load(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_aucs.npy")

# 自动统一 epoch 数量
epochs = min(train_losses.shape[1], valid_losses.shape[1], train_accs.shape[1], valid_accs.shape[1], auc.shape[1])
train_losses = train_losses[:, :epochs]
train_accs = train_accs[:, :epochs]
valid_losses = valid_losses[:, :epochs]
valid_accs = valid_accs[:, :epochs]
auc = auc[:, :epochs]

# 计算均值
mean_train_loss = np.mean(train_losses, axis=0)
mean_valid_loss = np.mean(valid_losses, axis=0)
mean_train_acc = np.mean(train_accs, axis=0)
mean_valid_acc = np.mean(valid_accs, axis=0)
mean_auc = np.mean(auc, axis=0)

# 设置字体与样式
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 2.5  # 加粗边框
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 如果需要嵌入字体，可以设置使用系统可识别的字体
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
titlefs = 20
legendfs = 20
lw = 3
xyfs = 20
tickfs = 20
# ---------------------- Loss 曲线 ----------------------
ax.set_title(f'{band} {connetc} Loss Curve', fontsize=titlefs, weight='bold')
# for i in range(10):
#     axs[0].plot(range(epochs), train_losses[i], color='blue', alpha=0.5)
#     axs[0].plot(range(epochs), valid_losses[i], color='red', alpha=0.5)
ax.plot(range(epochs), mean_train_loss, color='#00008B', label='Mean Train Loss', linewidth=lw)
ax.plot(range(epochs), mean_valid_loss, color='#8B0000', label='Mean Valid Loss', linewidth=lw)
ax.set_xlabel('Epoch', fontsize=xyfs, weight='bold')
ax.set_ylabel('Loss', fontsize=xyfs, weight='bold')
ax.legend(loc='upper right', fontsize=legendfs)
ax.tick_params(axis='both', labelsize=tickfs)  # 放大刻度字体

# # ---------------------- Accuracy 曲线 ----------------------
# axs[1].set_title('Accuracy Curve', fontsize=titlefs)
# for i in range(10):
#     axs[1].plot(range(epochs), train_accs[i], color='green', alpha=0.5)
#     axs[1].plot(range(epochs), valid_accs[i], color='orange', alpha=0.5)
# axs[1].plot(range(epochs), mean_train_acc, color='#006400', label='Mean Train Accuracy', linewidth=lw)
# axs[1].plot(range(epochs), mean_valid_acc, color='#FFA500', label='Mean Valid Accuracy', linewidth=lw)
# axs[1].set_xlabel('Epoch', fontsize=xyfs)
# axs[1].set_ylabel('Accuracy', fontsize=xyfs)
# axs[1].legend(loc='lower right', fontsize=legendfs)
# axs[1].set_xlabel('Epoch', fontsize=tickfs)
# axs[1].set_ylabel('Accuracy', fontsize=tickfs)
# axs[1].tick_params(axis='both', labelsize=tickfs)
#
# # ---------------------- AUC 曲线 ----------------------
# axs[2].set_title('AUC Curve', fontsize=titlefs)
# for i in range(10):
#     axs[2].plot(range(epochs), auc[i], color='#D8B9D6', alpha=0.5)
# axs[2].plot(range(epochs), mean_auc, color='#4B0082', label='Mean Valid AUC', linewidth=lw)
# axs[2].set_xlabel('Epoch', fontsize=xyfs)
# axs[2].set_ylabel('AUC', fontsize=xyfs)
# axs[2].legend(loc='upper right', fontsize=legendfs)
# axs[2].set_xlabel('Epoch', fontsize=tickfs)
# axs[2].set_ylabel('AUC', fontsize=tickfs)
# axs[2].tick_params(axis='both', labelsize=tickfs)

# 调整布局显示
plt.tight_layout()
plt.savefig(f"fig/loss_{band}_{connetc}.pdf", bbox_inches='tight')
plt.savefig(f"fig/loss_{band}_{connetc}.png", dpi=300)
plt.close()
