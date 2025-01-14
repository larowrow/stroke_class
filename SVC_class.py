from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
# 假设 X 是特征矩阵，y 是标签
rootPath = "data"
time_w = "time_8"
band = 'Gamma'
connetc = "pli"

resultpath = rootPath + "/" + time_w + "/" + band + "/"
# 31*15的特征
feature = np.load(resultpath + f'{connetc}_net_time_freq.npy')
y = np.load(resultpath + 'y_label.npy', allow_pickle=True)

# 分割数据为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'C': [10**-3, 10**-2, 10**-1, 1, 10, 10**2, 10**3],
    'gamma': [10**-3, 10**-2, 10**-1, 1, 10, 10**2]
}

# 初始化 SVM 模型
svm = SVC()

# 使用 GridSearchCV 进行网格搜索
grid_search = GridSearchCV(svm, param_grid, cv=10, scoring='accuracy', verbose=1)

# 在训练集上拟合模型
grid_search.fit(X_train, y_train)

# 输出最佳超参数
print("最佳超参数: ", grid_search.best_params_)

# 使用最佳参数训练整个训练集
best_svm = grid_search.best_estimator_

# 在测试集上评估模型
y_pred = best_svm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

# 计算 F1 分数
f1 = f1_score(y_test, y_pred)

# 计算精确度
precision = precision_score(y_test, y_pred)

# 计算召回率
recall = recall_score(y_test, y_pred)

# 输出各项评估指标
print(f"测试集准确率: {accuracy:.4f}")
print(f"F1 分数: {f1:.4f}")
print(f"精确度 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")

# 可以使用 classification_report 获取详细的评估报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred))