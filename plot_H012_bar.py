import json
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
# === 路径设置 ===
rootPath = "data"
time_three = 8.0
json_path = os.path.join(rootPath, f"time_{int(time_three)}", "band_homology_counts_plv.json")

# === 加载 JSON 文件 ===
with open(json_path, "r") as f:
    band_homology_counts = json.load(f)

# === 提取数据 ===
labels = list(band_homology_counts.keys())
H0_counts = [band_homology_counts[b]["H0"] for b in labels]
H1_counts = [band_homology_counts[b]["H1"] for b in labels]
H2_counts = [band_homology_counts[b]["H2"] for b in labels]

x = np.arange(len(labels))
width = 0.25
xyfont = 18
titlesize = 20
numfont = 14
# === 设置字体为 Times New Roman ===
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 如果需要嵌入字体，可以设置使用系统可识别的字体
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
# === 画图 ===
plt.figure(figsize=(10, 7))  # 加长纵轴

plt.bar(x - width, H0_counts, width, label='H0 (components)', color='C0')
plt.bar(x,         H1_counts, width, label='H1 (loops)', color='C1')
plt.bar(x + width, H2_counts, width, label='H2 (voids)', color='C2')

# === 轴标签与标题 ===
plt.xlabel("Frequency Band", fontsize=xyfont, weight='bold')
plt.ylabel("Number of Samples", fontsize=xyfont, weight='bold')
plt.title("PLV: Significant Persistent Homology Counts per Frequency Band", fontsize=titlesize, weight='bold')

# === 刻度字号 ===
plt.xticks(x, labels, fontsize=numfont)
plt.yticks(fontsize=numfont)
plt.ylim(0, 700)  # 将 y 轴范围设置为 0 到 600，根据你的需求修改数值

# === 图例 ===
plt.legend(fontsize=numfont, loc='upper right')

# === 网格线 ===
plt.grid(axis='y', linestyle='--', alpha=0.5)

# === 加粗边框线 ===
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

plt.tight_layout()

plt.savefig("H012_plv", dpi=300)
plt.savefig("H012_plv.pdf", format='pdf', bbox_inches='tight')
plt.show()