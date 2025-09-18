import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体和边框
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 参数设置
root_dir = "data/time_8"
bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']
methods = ['pcc', 'plv']
band_colors = {
    'Alpha': 'tab:blue',
    'Beta': 'tab:orange',
    'Delta': 'tab:green',
    'Gamma': 'tab:red',
    'Theta': 'tab:purple'
}

# 字体参数
legend_fontsize = 16
axis_fontsize = 16
title_fontsize = 20

for method in methods:
    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for band in bands:
        tprs, aucs = [], []

        for fold in range(10):
            if fold == 7 and band == 'Delta':
                continue

            base = os.path.join(root_dir, band, "test_result")
            try:
                fpr = np.load(os.path.join(base, f"{method}_fpr_kf_{fold}.npy"))
                tpr = np.load(os.path.join(base, f"{method}_tpr_kf_{fold}.npy"))
                auc = np.load(os.path.join(base, f"{method}_auc_kf_{fold}.npy"))[0]
            except:
                continue

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(auc)

            # 每一折曲线（浅色透明）
            plt.plot(mean_fpr, tpr_interp, color=band_colors[band], alpha=0.2, lw=1)

        if not tprs:
            continue

        # 计算平均曲线和标准差
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # 阴影区间 ±1 std
        plt.fill_between(mean_fpr,
                         np.maximum(mean_tpr - std_tpr, 0),
                         np.minimum(mean_tpr + std_tpr, 1),
                         color=band_colors[band], alpha=0.15)

        # 平均曲线（深色）
        plt.plot(mean_fpr, mean_tpr, color=band_colors[band], lw=2.5,
                 label=f"{band} (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

    # 图像设置
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate", fontsize=axis_fontsize)
    plt.ylabel("True Positive Rate", fontsize=axis_fontsize)
    plt.title(f"Mean ROC Curves ({method.upper()})", fontsize=title_fontsize, weight='bold')
    plt.legend(loc='lower right', fontsize=legend_fontsize)
    plt.grid(alpha=0.3)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"fig/ROC_{method.upper()}.pdf", bbox_inches='tight')
    plt.savefig(f"fig/ROC_{method.upper()}.png", dpi=300)
    print(f"保存成功：ROC_{method.upper()}")

    plt.close()
