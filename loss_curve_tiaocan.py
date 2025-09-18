

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# 配置参数
band = 'Theta'
connetc = "plv"
lr_list = [0.0005, 0.001, 0.005]#
dropoutp_list = [0.4, 0.5, 0.6, 0.7]
lr_step = 50
epochs_expect = 150  # 预期的 epochs 数量
weight_decay = 1e-4

for lr in lr_list:
    for dropoutp in dropoutp_list:
        # 加载数据
        train_losses = np.load(f"data/tiaocan/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_losses.npy")
        train_accs = np.load(f"data/tiaocan/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_accs.npy")
        valid_losses = np.load(f"data/tiaocan/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_losses.npy")
        valid_accs = np.load(f"data/tiaocan/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_accs.npy")
        auc = np.load(f"data/tiaocan/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_aucs.npy")


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
        ax.set_title(f'lr={lr} dropoutp={dropoutp} Loss Curve', fontsize=titlefs, weight='bold')
        # for i in range(10):
        #     axs[0].plot(range(epochs), train_losses[i], color='blue', alpha=0.5)
        #     axs[0].plot(range(epochs), valid_losses[i], color='red', alpha=0.5)
        ax.plot(range(epochs), mean_train_loss, color='#00008B', label='Mean Train Loss', linewidth=lw)
        ax.plot(range(epochs), mean_valid_loss, color='#8B0000', label='Mean Valid Loss', linewidth=lw)
        ax.set_xlabel('Epoch', fontsize=xyfs)
        ax.set_ylabel('Loss', fontsize=xyfs)
        ax.legend(loc='upper right', fontsize=legendfs)
        ax.tick_params(axis='both', labelsize=tickfs)  # 放大刻度字体

        # 调整布局显示
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"fig/tiaocan/Theta/loss_{band}_{connetc}_{lr}_{dropoutp}.pdf", bbox_inches='tight')
        plt.savefig(f"fig/tiaocan/Theta/loss_{band}_{connetc}_{lr}_{dropoutp}.png", dpi=300)
        plt.close()
