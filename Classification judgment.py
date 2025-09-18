import pandas as pd

# 读取数据
df = pd.read_csv("participants.tsv", sep="\t")

# 定义分组函数
# def classify(row):
#     if row["NIHSS"] <= 4 and row["mRS"] <= 2 and row["MBI"] >= 75:
#         return "Mild"
#     elif row["NIHSS"] >= 5 and row["mRS"] >= 3 and row["MBI"] < 75:
#         return "Moderate"
#     else:
#         return "Unclassified"
def classify(row):
    if row["NIHSS"] <= 4:
        return "Mild"
    elif row["NIHSS"] >= 5 and row["NIHSS"] <= 15:
        return "Moderate"
    else:
        return "Unclassified"
# 应用分组规则
df["Group"] = df.apply(classify, axis=1)

# 只输出 Participant_ID 和分组
result = df[["Participant_ID", "Group"]]

# 保存结果
# result.to_csv("participants_grouped.tsv", sep="\t", index=False)

print(result)
