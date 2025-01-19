import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取原始数据
raw = mne.io.read_raw_fif('data/severe/sub-01_task-motor-imagery_eeg.fif', preload=True)

# 删除不需要的电极通道
channels_to_drop = ['HEOL', 'HEOR', raw.ch_names[-1]]  # 根据你的数据调整
remaining_channels = [ch for ch in raw.ch_names if ch not in channels_to_drop]
raw.drop_channels(channels_to_drop)

# 设置电极排列为标准的10-20系统
electrode_data = pd.read_csv('task-motor-imagery_electrodes.tsv', sep='\t')
ch_pos = {row['name']: [row['X'], row['Y'], row['Z']] for _, row in electrode_data.iterrows()}

# 检查通道和电极名匹配
missing_channels = [ch for ch in electrode_data['name'] if ch not in raw.ch_names]
if missing_channels:
    print(f"警告：以下电极不在原始数据中：{missing_channels}")

# 创建自定义布局
montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
raw.set_montage(montage, on_missing='ignore')  # 忽略缺失的电极

# 加载激活值数据
file_path = 'CycleRatio_data/cycle_ratio_patients_delta_severe.txt'
data = np.loadtxt(file_path)

# 按第一列排序
sorted_data = data[data[:, 0].argsort()]  # 按第一列排序

# 提取排序后的第二列
activation_values = sorted_data[:, 1].tolist()

# 确保激活值数量与剩余电极数匹配
if len(activation_values) != len(remaining_channels):
    raise ValueError("激活值的数量与剩余电极的数量不匹配。")

# 绘制热力图
fig, ax = plt.subplots(figsize=(6, 6))
mne.viz.plot_topomap(activation_values, raw.info, axes=ax,
                     vlim=(min(activation_values), max(activation_values)), cmap='plasma',
                     show=False)

# 添加颜色条
mappable = plt.cm.ScalarMappable(cmap='plasma')
mappable.set_array(activation_values)
mappable.set_clim(vmin=min(activation_values), vmax=max(activation_values))  # 设置颜色条范围

# 创建颜色条
colorbar = plt.colorbar(mappable, ax=ax)

# 设置颜色条刻度字体大小
colorbar.ax.tick_params(labelsize=16)  # 设置刻度标签字体大小
# 设置颜色条刻度字体为 Times New Roman
for label in colorbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')  # 设置字体
    label.set_fontweight('bold')  # 设置加粗
# 设置颜色条标签字体大小和字体样式
colorbar.set_label('Cycle number', fontsize=20, fontname='Times New Roman', fontweight='bold')  # 设置标签的字体和样式

# 设置颜色条刻度线加粗
colorbar.ax.tick_params(width=2, length=6)

# 设置颜色条边框加粗
for spine in colorbar.ax.spines.values():
    spine.set_linewidth(2)  # 设置边框粗细

# 保存图像
filename = '_'.join(file_path.split('_')[-2:])  # 获取倒数第二和第三部分并拼接
filename = filename.replace('.txt', '2.png')  # 替换扩展名
plt.savefig(filename, dpi=300)  # 保存图像
plt.show()

