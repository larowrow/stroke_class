#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# 加载数据
rootPath = "data"
time_w = "time_8"
band = "Gamma"
connetc = "pli"

resultpath = f"{rootPath}/{time_w}/{band}/"
feature = np.load(resultpath + f"{connetc}_net_time_freq.npy")
y = np.load(resultpath + "y_label.npy", allow_pickle=True)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.3, random_state=42, stratify=y)

# 定义随机森林模型和超参数范围
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [20, 50, 100, 150],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [10, 15, 20, 30],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None]
}

# 网格搜索
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 获取最佳参数和最佳模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("最佳参数:", best_params)

# 使用最佳模型预测
y_pred = best_model.predict(X_test)

# 评估性能
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 输出分类报告，保留四位小数
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))  # 保留四位小数
