import os
import numpy as np

# 设置数据目录和子文件夹
base_dir = "data/time_8"
subfolders = ["Alpha", "Beta", "Delta", "Theta", "Gamma"]

# 特征提取索引（Python 是从 0 开始索引的）
# 即第 1,2,5,7,8-15 对应 0,1,4,6,7~14
feature_indices = [0, 1, 4, 6] + list(range(7, 15))

# 通道数和每个通道原始特征数
n_channels = 29
features_per_channel = 15

# 构造需要保留的总特征索引（在 435 维特征中的位置）
selected_indices = []
for c in range(n_channels):
    base = c * features_per_channel
    selected_indices.extend([base + i for i in feature_indices])

# 遍历每个子文件夹
for folder in subfolders:
    folder_path = os.path.join(base_dir, folder)

    for prefix in ["plv", "pcc"]:
        filename = f"{prefix}_net_time_freq.npy"
        filepath = os.path.join(folder_path, filename)

        # 加载数据
        data = np.load(filepath)  # shape: (720, 435)

        # 选取所需特征
        selected_data = data[:, selected_indices]

        # 保存新文件
        save_name = f"MI_{filename}"
        save_path = os.path.join(folder_path, save_name)
        np.save(save_path, selected_data)

        print(f"Saved: {save_path}, shape: {selected_data.shape}")
