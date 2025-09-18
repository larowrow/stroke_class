import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, mannwhitneyu, fisher_exact, shapiro
import numpy as np

# === Step 1: 读取数据 ===
df = pd.read_csv("participants.tsv", sep="\t")

# === Step 2: 添加分组标签 ===
mild_indices = [i - 1 for i in [3,7,8,12,14,16,17,20,21,28,29,30,34,44,46,48,49,50]]
moderate_indices = [i - 1 for i in [1,11,13,18,19,22,25,36,39,40,41,42,47]]

df["Group"] = "Other"
df.loc[mild_indices, "Group"] = "Mild"
df.loc[moderate_indices, "Group"] = "Moderate"
df = df[df["Group"].isin(["Mild", "Moderate"])]  # 去除未分组样本

# === Step 3: 提取病灶侧 ===
def extract_side(text):
    text = str(text).lower()
    if 'left' in text and 'right' in text:
        return 'bilateral'
    elif 'left' in text:
        return 'left'
    elif 'right' in text:
        return 'right'
    else:
        return 'bilateral'

df["LesionSide"] = df["StrokeLocation"].apply(extract_side)

# === Step 4: 连续变量比较 ===
def compare_continuous(var):
    group1 = df[df["Group"] == "Mild"][var].dropna()
    group2 = df[df["Group"] == "Moderate"][var].dropna()

    # 检查正态性
    p_norm1 = shapiro(group1)[1]
    p_norm2 = shapiro(group2)[1]

    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(), group2.std()

    if p_norm1 > 0.05 and p_norm2 > 0.05:  # 正态
        stat, pval = ttest_ind(group1, group2, equal_var=False)
        test_name = "Welch's t-test"
    else:  # 非正态
        stat, pval = mannwhitneyu(group1, group2, alternative="two-sided")
        test_name = "Mann-Whitney U"
    print(f'shapiro: p1={p_norm1} and p2={p_norm2}')

    print(f"{var} ({test_name}):")
    print(f"  Mild: {mean1:.2f} ± {std1:.2f}, Moderate: {mean2:.2f} ± {std2:.2f}")
    print(f"  stat = {stat:.3f}, p = {pval:.4f}\n")

print("=== 连续变量比较 ===")
for col in ["Age", "Duration", "NIHSS", "MBI", "mRS"]:
    compare_continuous(col)

# === Step 5: 类别变量比较 ===
def compare_categorical(var):
    table = pd.crosstab(df[var], df["Group"])
    chi2, pval, dof, expected = chi2_contingency(table)

    if (expected < 5).any():  # 理论频数过低 → Fisher精确检验（仅限2x2）
        if table.shape == (2, 2):
            _, pval = fisher_exact(table)
            test_name = "Fisher's Exact Test"
        else:
            test_name = "Chi-square (注意: 部分格子频数<5)"
    else:
        test_name = "Chi-square test"

    print(f"{var} ({test_name}):")
    print(table)
    print(f"  p = {pval:.4f}\n")

# === Step 6: 粗略解剖分区分类 ===
def classify_region(location: str) -> str:
    """根据StrokeLocation粗略划分脑区"""
    text = str(location).lower()
    if any(k in text for k in ["pons", "medulla", "brainstem"]):
        return "Brainstem"
    elif any(k in text for k in ["basal ganglia", "paraventricular", "thalamus", "subcortical"]):
        return "BasalGanglia_Subcortical"
    elif "cerebell" in text:
        return "Cerebellum"
    elif any(k in text for k in ["frontal", "parietal", "temporo", "cortex"]):
        return "Cortex"
    else:
        return "Other"

df["LesionRegion"] = df["StrokeLocation"].apply(classify_region)

# === Step 7: 分组比较 ===
def compare_region(var="LesionRegion"):
    table = pd.crosstab(df[var], df["Group"])
    chi2, pval, dof, expected = chi2_contingency(table)

    if (expected < 5).any():
        test_name = "Chi-square (注意: 部分格子频数<5)"
    else:
        test_name = "Chi-square test"

    print(f"{var} ({test_name}):")
    print(table)
    print(f"  p = {pval:.4f}\n")

print("=== 解剖分区比较 ===")
compare_region()


print("=== 类别变量比较 ===")
for col in ["Gender", "ParalysisSide", "IsFirstTime", "LesionSide"]:
    compare_categorical(col)


