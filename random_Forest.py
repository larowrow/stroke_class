import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, RocCurveDisplay, make_scorer, f1_score, recall_score, precision_score

# 加载数据
rootPath = "data"
time_w = "time_8"
band = "Gamma"
connetc = "plv"

resultpath = f"{rootPath}/{time_w}/{band}/"
feature = np.load(resultpath + f"MI_{connetc}_net_time_freq.npy")
y = np.load(resultpath + "y_label.npy", allow_pickle=True)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=42)

# 定义随机森林模型和超参数范围
rf = RandomForestClassifier(random_state=42, n_estimators=185)  # 直接使用已知最佳参数

# 定义评估指标
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted'),
    'precision': make_scorer(precision_score, average='weighted')
}

# 十折交叉验证
cv_results = cross_validate(rf, X_train, y_train, cv=10, scoring=scoring, n_jobs=-1, verbose=2)

# 计算平均值和标准差
avg_accuracy = np.mean(cv_results['test_accuracy'])
std_accuracy = np.std(cv_results['test_accuracy'])

avg_f1 = np.mean(cv_results['test_f1'])
std_f1 = np.std(cv_results['test_f1'])

avg_recall = np.mean(cv_results['test_recall'])
std_recall = np.std(cv_results['test_recall'])

avg_precision = np.mean(cv_results['test_precision'])
std_precision = np.std(cv_results['test_precision'])
# 输出十折交叉验证结果
# 输出十折交叉验证结果
print("\n十折交叉验证性能:")
print(f"Accuracy : {avg_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"F1-score : {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Recall   : {avg_recall:.4f} ± {std_recall:.4f}")
print(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}")
# 训练最终模型
rf.fit(X_train, y_train)

# 测试集评估
y_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n测试集性能:")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
# # 计算ROC曲线指标
# y_proba = best_model.predict_proba(X_test)[:, 1]  # 取正类的概率
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# roc_auc = auc(fpr, tpr)
# # 绘制ROC曲线
# plt.figure(figsize=(8, 6))
# RocCurveDisplay.from_estimator(
#     best_model,
#     X_test,
#     y_test,
#     name=f"Random Forest (AUC = {roc_auc:.2f})",
#     color="darkorange",
# )
# plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title(f"ROC Curve - {band} Band (PLV)")
# plt.legend(loc="lower right")
# plt.show()
#
# # 检查FPR=0附近的点
# fpr, tpr, thresholds = roc_curve(y_test, y_proba)
# start_idx = np.where(fpr <= 0.01)[0]  # 取FPR接近0的点
# print("FPR≈0时的TPR:", tpr[start_idx])
# print("对应阈值:", thresholds[start_idx])
# from sklearn.metrics import confusion_matrix
# # 选择FPR≈0时的阈值
# optimal_idx = np.argmin(fpr)
# optimal_threshold = thresholds[optimal_idx]
# y_pred = (y_proba >= optimal_threshold).astype(int)
# print(confusion_matrix(y_test, y_pred))
# import seaborn as sns
# sns.boxplot(x=y_test, y=y_proba)
# plt.title("pre pro distri")
# plt.show()
#
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,4))
# plt.subplot(121)
# plt.hist(y_proba[y_test==0], bins=20, alpha=0.5, label='Class 0')
# plt.hist(y_proba[y_test==1], bins=20, alpha=0.5, label='Class 1')
# plt.legend()
# plt.subplot(122)
# plt.scatter(range(len(y_proba)), y_proba, c=y_test, cmap='bwr')
# plt.colorbar(label='True Label')
# plt.show()
#
# import pandas as pd
# proba_stats = pd.DataFrame({
#     'y_true': y_test,
#     'y_proba': y_proba
# })
# print(proba_stats.groupby('y_true')['y_proba'].describe())
#
# # 检查训练/测试集特征分布
# plt.figure(figsize=(12,4))
# plt.subplot(131)
# plt.hist(X_train[:,0], bins=30, alpha=0.5, label='Train')
# plt.hist(X_test[:,0], bins=30, alpha=0.5, label='Test')
# plt.legend()
# plt.subplot(132)
# plt.scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.2)
# plt.title('Train')
# plt.subplot(133)
# plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker='x')
# plt.title('Test')
# plt.show()
#%% select number of trees
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


# # 加载数据
# rootPath = "data"
# time_w = "time_8"
# band = "Gamma"
# connetc = "plv"
#
# resultpath = f"{rootPath}/{time_w}/{band}/"
# feature = np.load(resultpath + f"MI_{connetc}_net_time_freq.npy")
# y = np.load(resultpath + "y_label.npy", allow_pickle=True)
#
# # 数据集划分
# X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.3, random_state=42, stratify=y)
# oob_errors = []
# for n_estimators in range(50, 301, 27):
#     model = BaggingRegressor(base_estimator=DecisionTreeRegressor(random_state=123),
#                              n_estimators=n_estimators, n_jobs=-1, oob_score=True, random_state=0)
#     model.fit(X_train, y_train)
#     pred_oob = model.oob_prediction_
#     oob_errors.append(mean_squared_error(y_train, pred_oob))
# np.save('oob_errors.npy', oob_errors)
#
# # 定义绘图参数
# label_size = 18
# spines_width = 2
# font_size_ticks = 15
# ticks_width = 2.0
# # 设置全局字体样式和大小
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['axes.labelweight'] = 'bold'
# plt.rcParams['xtick.labelsize'] = font_size_ticks
# plt.rcParams['ytick.labelsize'] = font_size_ticks
# plt.rcParams['axes.labelsize'] = label_size
# fig, ax = plt.subplots(figsize=(5.7, 5.1))  # 设置图形大小
# plt.plot(range(50, 301, 27), oob_errors)
# plt.xlabel('Number of Trees', fontsize=label_size, fontname='Times New Roman', fontweight='bold')
# plt.ylabel('OOB MSE', fontsize=label_size, fontname='Times New Roman', fontweight='bold')
# # plt.title('Bagging OOB Errors')
# for axis in ['top', 'bottom', 'left', 'right']:
#     ax.spines[axis].set_linewidth(spines_width)  # 边框粗细
# # 设置纵轴刻度线属性
# ax.tick_params(axis='both', which='major', direction='out', width=ticks_width, labelsize=font_size_ticks)
# plt.tight_layout()
# plt.savefig("fig/RF_OOBmse.png", dpi=300, bbox_inches='tight')
# plt.savefig("fig/RF_OOBmse.pdf", dpi=300, bbox_inches='tight')
# plt.show()