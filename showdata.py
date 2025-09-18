import mne

# 读取原始 .fif 文件
file_path = 'data/mild/sub-03_task-motor-imagery_eeg.fif'
raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)

# 获取所有通道名称
channel_names = raw.info['ch_names']

# 打印通道列表
print("通道总数：", len(channel_names))
print("通道名称列表：")
for name in channel_names:
    print(name)

import numpy as np
import matplotlib.pyplot as plt
from gtda import pipeline
from scipy.signal import hilbert
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from sklearn.pipeline import make_pipeline
import pandas as pd
import mne
from mne.filter import filter_data
import os
from collections import Counter

# ===== 可视化前几个样本的条形码 =====
# def plot_persistence_barcodes(diagrams, n_plot=5):
#     fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2.5 * n_plot), sharex=True)
#     if n_plot == 1:
#         axes = [axes]
#
#     for i, ax in enumerate(axes):
#         ax.set_title(f"Sample {i + 1} Barcode Diagram")
#         for dim in np.unique(diagrams[i][:, 2]):
#             for birth, death in diagrams[i][diagrams[i][:, 2] == dim][:, 0:2]:
#                 ax.hlines(y=dim + 0.1 * np.random.rand(), xmin=birth, xmax=death,
#                           colors=f'C{int(dim)}', linewidth=2)
#         ax.set_ylabel("Homology\nDimension")
#         ax.set_ylim(-0.5, 2.5)
#     plt.xlabel("Filtration Value")
#     plt.tight_layout()
#     plt.show()

# ===== 主流程 =====
rootPath = "data"
time_three = 8.0
bands = {
    "Delta": [1, 3],
    "Theta": [4, 7],
    "Alpha": [8, 13],
    "Beta": [14, 30],
    "Gamma": [31, 40]
}

# 阈值：滤除短生命周期的噪声
persistence_threshold = 0.05

# 创建 TDA pipeline
tda_pipeline = make_pipeline(
    PearsonDissimilarity(),
    VietorisRipsPersistence(n_jobs=-1)
)
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
            sample = window_data.T  # shape: (4000, 29)
            samples.append(sample)

    print(f"  Total Samples in {band_name}: {len(samples)}")

    # === 计算所有样本的持久图 ===
    diagrams_all = []
    for sample in samples:
        sample_reshaped = sample[np.newaxis, :, :]
        diagram = tda_pipeline.fit_transform(sample_reshaped)
        diagrams_all.append(diagram[0])

    # === 可视化前 n_plot 个条形码 ===
    # plot_persistence_barcodes(diagrams_all, n_plot=5)

    # === 输出所有样本的 H0/H1/H2 统计 ===
    # === 初始化全局样本数量统计 ===
    count_H0_samples = 0
    count_H1_samples = 0
    count_H2_samples = 0

    print("  Homology Group Counts (filtered by persistence threshold):")
    for i, diag in enumerate(diagrams_all):
        homology_counts = {0: 0, 1: 0, 2: 0}

        for birth, death, dim in diag:
            if death - birth >= persistence_threshold and int(dim) in homology_counts:
                homology_counts[int(dim)] += 1

        # 样本级别统计：只要某个维度有至少一个特征，就加一
        if homology_counts[0] > 0:
            count_H0_samples += 1
        if homology_counts[1] > 0:
            count_H1_samples += 1
        if homology_counts[2] > 0:
            count_H2_samples += 1

    # === 最终样本计数结果 ===
    print("\n--- 样本中含有 H0/H1/H2 的数量（已过滤噪声）---")
    print(f"  含 H0 的样本数: {count_H0_samples}")
    print(f"  含 H1 的样本数: {count_H1_samples}")
    print(f"  含 H2 的样本数: {count_H2_samples}")

    # === 保存当前频段统计到总字典 ===
    band_homology_counts[band_name] = {
        "H0": count_H0_samples,
        "H1": count_H1_samples,
        "H2": count_H2_samples,
        "Total": len(samples)
    }

import json

save_path = os.path.join(rootPath, f"time_{int(time_three)}", "band_homology_counts_pcc.json")
with open(save_path, "w") as f:
    json.dump(band_homology_counts, f, indent=4)

print(f"已保存到 {save_path}")
