import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import matplotlib as mpl
# === 配置 ===
rootPath = "data"
t = "time_8"
bands = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Theta']
netcs = ['plv', 'pcc']
xtickname = ['mav', 'std', 'skewness', 'kurtosis', 'betweenness', 'clustering', 'psd',
             'PersistenceEntropy_0', 'PersistenceEntropy_1', 'landscape1_0', 'landscape1_1',
             'landscape2_0', 'landscape2_1', 'betti_0', 'betti_1']
colors_band = ['skyblue', 'salmon', 'mediumseagreen', 'orange', 'mediumpurple']
colors_netc = ['cornflowerblue', 'lightcoral', 'mediumseagreen']

num_nodes = 29
features_per_node = 15

# === 初始化结果结构 ===
# shape: (netc_idx, band_idx, 15)
mi_all = np.zeros((len(netcs), len(bands), features_per_node))

# === 计算 MI ===
for netc_idx, netc in enumerate(netcs):
    for band_idx, band in enumerate(bands):
        resultpath = f"{rootPath}/{t}/{band}/"
        feature = np.load(resultpath + f'{netc}_net_time_freq.npy')  # (720, 435)
        label = np.load(resultpath + 'y_label.npy', allow_pickle=True)  # (720,)
        feature_3d = feature.reshape(-1, num_nodes, features_per_node)

        mi_matrix = np.zeros((num_nodes, features_per_node))
        for node_idx in range(num_nodes):
            mi_matrix[node_idx] = mutual_info_classif(feature_3d[:, node_idx, :], label, random_state=42)

        mean_mi = mi_matrix.mean(axis=0)  # (15,)
        mi_all[netc_idx, band_idx, :] = mean_mi


# === 图1: 每个特征显示不同 netc（每组3个柱子），每个柱子是5个频段的平均 ===
xyfont = 16
titlesize = 20
numfont = 16
mi_avg_band = mi_all.mean(axis=1)
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
bar_width = 0.2
x = np.arange(features_per_node)
num_groups = len(netcs)

plt.figure(figsize=(13, 6))
for i in range(num_groups):
    offset = (i - (num_groups - 1)/2) * bar_width  # 保持原有偏移计算
    plt.bar(x + offset, mi_avg_band[i], width=bar_width, label=netcs[i].upper(), color=colors_netc[i])

# 修改此处：调整刻度位置和对齐参数
plt.xticks(
    x,  # 关键：使用原始特征中心位置作为刻度
    xtickname,
    rotation=45,
    ha='right',
    rotation_mode='anchor',
    fontsize=numfont
)

plt.ylabel("Avg MI across Frequency Bands", fontsize=xyfont, fontweight='bold')
plt.xlabel("Feature", fontsize=xyfont, fontweight='bold')
plt.title("Feature Importance: Avg across Frequency Bands (Grouped by Functional Connectivity Metric)", fontsize=titlesize, fontweight='bold')
plt.legend(title='Functional Connectivity Metric', fontsize=xyfont)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("MI_feature_grouped_by_netc_fixed.pdf", bbox_inches='tight')
plt.savefig("MI_feature_grouped_by_netc_fixed.png", dpi=300, bbox_inches='tight')
plt.show()
# === 图2: 每个特征显示不同 band（每组5个柱子），每个柱子是3个 netc 的平均 ===
mi_avg_netc = mi_all.mean(axis=0)  # shape: (5, 10) → 每个 band 在所有 netc 上的平均
bar_width = 0.13
num_groups = len(bands)
x = np.arange(features_per_node)  # x 是每个特征中心点的位置

plt.figure(figsize=(13, 6))

for i in range(num_groups):
    offset = (i - (num_groups - 1) / 2) * bar_width
    plt.bar(x + offset, mi_avg_netc[i], width=bar_width, label=bands[i], color=colors_band[i])

# 修改此处：添加 ha 和 rotation_mode 参数
plt.xticks(x, xtickname, rotation=45, ha='right', rotation_mode='anchor', fontsize=numfont)
plt.ylabel("Avg MI across Functional Connectivity Metric", fontsize=xyfont, fontweight='bold')
plt.xlabel("Feature", fontsize=xyfont, fontweight='bold')
plt.title("Feature Importance: Avg across Functional Connectivity Metric (Grouped by Frequency Band)", fontsize=titlesize, fontweight='bold')
plt.legend(title='Frequency Band', fontsize=xyfont)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("MI_feature_grouped_by_band_fixed.pdf", bbox_inches='tight')
plt.savefig("MI_feature_grouped_by_band_fixed.png", dpi=300, bbox_inches='tight')
plt.show()

# for netc_idx, netc in enumerate(netcs):
#     for band_idx, band in enumerate(bands):
#         mi_vals = mi_all[netc_idx, band_idx]
#         sorted_idx = np.argsort(mi_vals)[::-1]
#         print(f"\n===== {netc.upper()} - {band} =====")
#         for rank, idx in enumerate(sorted_idx, 1):
#             print(f"{rank}. {xtickname[idx]} ({mi_vals[idx]:.4f})")
