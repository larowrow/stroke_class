import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mne.viz import plot_topomap
from mne.channels.layout import _find_topomap_coords
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# 如果需要嵌入字体，可以设置使用系统可识别的字体
mpl.rcParams['font.family'] = 'Times New Roman'  # 或其他常用字体
# 设置路径和频段
folder_path = 'CycleRatio_data'
frequencies = ['gamma', 'alpha', 'beta', 'delta', 'theta']
severities = ['mild', 'severe']

# 加载原始 EEG 数据和电极位置信息（只加载一次）
raw = mne.io.read_raw_fif('data/severe/sub-01_task-motor-imagery_eeg.fif', preload=True)

# 删除不需要的电极通道
channels_to_drop = ['HEOL', 'HEOR', raw.ch_names[-1]]  # 根据你的数据调整
remaining_channels = [ch for ch in raw.ch_names if ch not in channels_to_drop]
raw.drop_channels(channels_to_drop)

# 设置电极排列为标准的10-20系统
electrode_data = pd.read_csv('task-motor-imagery_electrodes.tsv', sep='\t')
ch_pos = {row['name']: [row['X'], row['Y'], row['Z']] for _, row in electrode_data.iterrows()}
montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
raw.set_montage(montage, on_missing='ignore')  # 忽略缺失电极

# ========== Step 1：收集所有文件中的最小最大值以统一 colorbar ==========
all_values = []

for severity in severities:
    for freq in frequencies:
        file_path = f'{folder_path}/cycle_ratio_patients_{freq}_{severity}.txt'
        data = np.loadtxt(file_path)
        sorted_data = data[data[:, 0].argsort()]
        activation_values = sorted_data[:, 1].tolist()

        if len(activation_values) != len(remaining_channels):
            raise ValueError(f"{file_path}：激活值数量与剩余通道数量不匹配！")

        all_values.extend(activation_values)

vmin, vmax = min(all_values), max(all_values)

# ========== Step 2：绘制 10 张图 ==========
for severity in severities:
    for freq in frequencies:
        file_path = f'{folder_path}/cycle_ratio_patients_{freq}_{severity}.txt'
        data = np.loadtxt(file_path)
        sorted_data = data[data[:, 0].argsort()]
        activation_values = sorted_data[:, 1].tolist()

        fig, ax = plt.subplots(figsize=(6, 6))

        # 获取通道位置并旋转（保持鼻子朝上）
        picks = mne.pick_types(raw.info, eeg=True)  # 只取 EEG 通道
        pos = _find_topomap_coords(raw.info, picks=picks)

        theta = np.deg2rad(90)  # 顺时针旋转 90°
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta),  np.cos(theta)]])
        pos_rotated = pos @ rotation_matrix.T

        # 绘制旋转后的 topomap
        plot_topomap(activation_values, pos_rotated, axes=ax,
                     vlim=(vmin, vmax), cmap='Spectral', show=False)

        # 添加标题
        ax.set_title(f'{severity}_{freq}', fontsize=20,
                     fontname='Times New Roman', fontweight='bold', pad=20)

        # 创建统一颜色条
        mappable = plt.cm.ScalarMappable(cmap='Spectral')
        mappable.set_array(activation_values)
        mappable.set_clim(vmin=vmin, vmax=vmax)
        colorbar = plt.colorbar(mappable, ax=ax)
        colorbar.ax.tick_params(labelsize=16)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
        colorbar.set_label('Occurrence count', fontsize=20,
                           fontname='Times New Roman', fontweight='bold')
        colorbar.ax.tick_params(width=2, length=6)
        for spine in colorbar.ax.spines.values():
            spine.set_linewidth(2)

        # plt.show()
        # 保存图像
        if not os.path.exists('fig'):
            os.makedirs('fig')

        filename = f'{freq}_{severity}'
        plt.savefig(f'fig/{filename}.png', dpi=300)
        plt.savefig(f'fig/{filename}.pdf', bbox_inches='tight')
        plt.close()
