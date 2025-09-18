import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 如果需要嵌入字体，可以设置使用系统可识别的字体
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
# ======================
# 1. 读入临床数据
# ======================
df = pd.read_csv("participants.tsv", sep="\t")

# ======================
# 2. 加入解码率（按你提供的顺序）
# ======================
decoding_rates = [
    69.5, 69.8, 66, 64, 65, 71, 80, 64, 67, 69.8,
    80, 71, 75, 69, 79, 62, 71, 71, 67, 81,
    75, 80, 76, 69, 75, 79.8, 69.9, 77, 69, 77,
    61, 68, 67, 64, 64, 65, 76, 75, 65, 80,
    60, 61, 61, 77, 81, 71, 70, 61, 74, 80
]
df["DecodingRate"] = decoding_rates


# ======================
# 3. 绘图函数
# ======================
def plot_scatter_with_regression(x, y, xlabel, ylabel, title, filename):
    plt.figure(figsize=(6, 5))
    sns.regplot(x=x, y=y, data=df, ci=None, scatter_kws={"s": 50}, line_kws={"color": "red"})

    # 计算相关系数
    r, p = pearsonr(df[x], df[y])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{title}\nPearson r={r:.2f}, p={p:.3f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.savefig(f"{filename}.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


# ======================
# 4. 生成三张图
# ======================
plot_scatter_with_regression("NIHSS", "DecodingRate",
                             "NIHSS", "Decoding Accuracy (%)",
                             "Decoding Rate vs NIHSS", "decoding_vs_NIHSS.png")

plot_scatter_with_regression("mRS", "DecodingRate",
                             "mRS", "Decoding Accuracy (%)",
                             "Decoding Rate vs mRS", "decoding_vs_mRS.png")

plot_scatter_with_regression("MBI", "DecodingRate",
                             "MBI", "Decoding Accuracy (%)",
                             "Decoding Rate vs MBI", "decoding_vs_MBI.png")
