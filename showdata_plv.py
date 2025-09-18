import numpy as np
from gtda.homology import VietorisRipsPersistence
import pandas as pd
import mne
from mne.filter import filter_data
import os
from scipy.signal import hilbert
import json
# ==================== 自定义 PLV 和 PLI ====================

def plv_matrix(X):
    ch = X.shape[1]
    PLV = np.ones((ch, ch))
    for ch1 in range(ch - 1):
        for ch2 in range(ch1 + 1, ch):
            PLV[ch1, ch2] = phase_locking_value(X[:, ch1], X[:, ch2])
            PLV[ch2, ch1] = PLV[ch1, ch2]
    return PLV

def phase_locking_value(theta1, theta2):
    theta1 = np.unwrap(np.angle(hilbert(theta1)))
    theta2 = np.unwrap(np.angle(hilbert(theta2)))
    complex_phase_diff = np.exp(1j * (theta1 - theta2))
    return np.abs(np.sum(complex_phase_diff)) / len(theta1)


# ==================== 主流程 ====================
rootPath = "data"
time_three = 8.0
bands = {
    "Delta": [1, 3],
    "Theta": [4, 7],
    "Alpha": [8, 13],
    "Beta": [14, 30],
    "Gamma": [31, 40]
}
persistence_threshold = 0.05
tda = VietorisRipsPersistence(metric="precomputed", n_jobs=-1)

band_homology_counts = {}
for band_name, band_range in bands.items():
    print(f"\nProcessing Band: {band_name}")
    resultpath = os.path.join(rootPath, f"time_{int(time_three)}", band_name)
    df = pd.read_csv(os.path.join(resultpath, "stroke_window_index.csv"))
    grouped_df = df.groupby("raw_file_path")
    samples = []

    for raw_file_path, group_df in grouped_df:
        print(f"  Reading File: {raw_file_path}")
        raw = mne.io.read_raw_fif(raw_file_path, preload=True, verbose=False)
        raw_data = raw.get_data(picks=np.arange(29))[:, :raw.n_times // 2]
        filtered_data = filter_data(raw_data, sfreq=raw.info['sfreq'],
                                    l_freq=band_range[0], h_freq=band_range[1])

        for window_idx in group_df.index.tolist():
            start_sample = group_df.loc[window_idx]['start_sample_index']
            stop_sample = group_df.loc[window_idx]['end_sample_index']
            window_data = filtered_data[:, start_sample:stop_sample]
            sample = window_data.T
            samples.append(sample) #sample(timepoint,channel)

    print(f"  Total Samples in {band_name}: {len(samples)}")

    # === 构造距离矩阵 + 计算持久图 ===
    diagrams_all = []
    for sample in samples:
        dist_matrix = 1 - plv_matrix(sample)  # 使用 PLV：越相似越小
        dist_matrix = np.clip(dist_matrix, 0, 1)  # 保证合法
        dist_matrix = dist_matrix[np.newaxis, :, :]  # shape: (1, n_channels, n_channels)
        diagram = tda.fit_transform(dist_matrix)
        diagrams_all.append(diagram[0])

    # === 统计每个维度的拓扑特征数量（过滤短生命期）===
    count_H0_samples = 0
    count_H1_samples = 0
    count_H2_samples = 0
    persistence_H0 = []
    persistence_H1 = []

    for diag in diagrams_all:
        homology_counts = {0: 0, 1: 0, 2: 0}
        for birth, death, dim in diag:
            persistence = death - birth
            if persistence >= persistence_threshold:
                if int(dim) == 0:
                    homology_counts[0] += 1
                    persistence_H0.append(persistence)
                elif int(dim) == 1:
                    homology_counts[1] += 1
                    persistence_H1.append(persistence)
                elif int(dim) == 2:
                    homology_counts[2] += 1

        if homology_counts[0] > 0:
            count_H0_samples += 1
        if homology_counts[1] > 0:
            count_H1_samples += 1
        if homology_counts[2] > 0:
            count_H2_samples += 1

    print("\n--- 样本中含有 H0/H1/H2 的数量（已过滤噪声）---")
    print(f"  含 H0 的样本数: {count_H0_samples}")
    print(f"  含 H1 的样本数: {count_H1_samples}")
    print(f"  含 H2 的样本数: {count_H2_samples}")

    # print("\n--- H0 和 H1 的 Persistence 统计（death - birth）---")
    # if persistence_H0:
    #     print(f"  H0 持久性数量: {len(persistence_H0)}")
    #     print(f"  H0 平均持久性: {np.mean(persistence_H0):.4f}")
    #     print(f"  H0 最大持久性: {np.max(persistence_H0):.4f}")
    # if persistence_H1:
    #     print(f"  H1 持久性数量: {len(persistence_H1)}")
    #     print(f"  H1 平均持久性: {np.mean(persistence_H1):.4f}")
    #     print(f"  H1 最大持久性: {np.max(persistence_H1):.4f}")

    band_homology_counts[band_name] = {
        "H0": count_H0_samples,
        "H1": count_H1_samples,
        "H2": count_H2_samples,
        "Total": len(samples)
    }

# === 保存结果 ===
save_path = os.path.join(rootPath, f"time_{int(time_three)}", "band_homology_counts_plv.json")
with open(save_path, "w") as f:
    json.dump(band_homology_counts, f, indent=4)
print(f"已保存到 {save_path}")
