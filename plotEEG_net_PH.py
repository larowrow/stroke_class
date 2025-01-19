import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.serif'] = ['Times New Roman']

# 定义绘图参数
font_size_ticks = 15
ticks_length = 10
parameter_size = 15
label_size = 25
legend_size = 16
title_size = 14
legendlength = 0.2
spines_width = 2
ticks_length_minor = 6
handletextpad = 0.5
ticks_width = 2.0
markersize2 = 10

# 假设 full_path 已经按照你的代码定义好
full_path = 'edffile0\\sub-01_task-motor-imagery_eeg.fif'
raw = mne.io.read_raw_fif(full_path, preload=True)

# 获取通道名称、采样频率以及去掉最后一个通道的数据
ch_names = raw.info['ch_names'][:-1]
sfreq = raw.info['sfreq']
data = raw.get_data()[:-1]  # 去掉最后一个通道的数据

# 计算15s和30s对应的时间索引
start_sample = int(15 * sfreq)  # 开始样本点索引
end_sample = int(30 * sfreq)    # 结束样本点索引

# 对数据进行时间上的切片
timeslice_data = data[:, start_sample:end_sample]

# 创建时间轴，从15秒开始
times = np.arange(start_sample, end_sample) / sfreq - start_sample / sfreq + 15

# 创建新的图形和轴
fig, ax1 = plt.subplots(figsize=(12, 8))  # 使用 subplots 方法创建图形和轴

# 设置轴线宽度
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(spines_width)
# 设置纵轴刻度线的属性
ax1.tick_params(axis='y', which='both', direction='out', width=ticks_width, length=ticks_length)
# 每个通道的数据都画一条线
for idx, ch_name in enumerate(ch_names):
    ax1.plot(times, timeslice_data[idx] + idx * np.exp(-10), label=ch_name,lw=2)  # 加上偏移量以区分不同通道

# 自定义纵轴标签
ax1.set_yticks(np.arange(len(ch_names)) * np.exp(-10))
ax1.set_yticklabels(ch_names, fontsize=font_size_ticks, fontweight='bold',fontname='Times New Roman')

# 移除横轴刻度和标签
ax1.set_xticks([])  # 清空横轴刻度

# 调整布局以适应所有标签
plt.tight_layout()

# 显示图表
plt.show()

# 如果你想保存图片，请取消下一行的注释
fig.savefig("eeg_allchannel.png", dpi=300, bbox_inches='tight')

#%%
##### EEG_net
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.io import loadmat

# 设置全局字体样式和大小
plt.rcParams['font.family'] = 'serif'  # 使用 serif 类型字体
plt.rcParams['font.serif'] = ['Times New Roman']  # 设置默认字体为 Times New Roman
plt.rcParams['axes.labelweight'] = 'bold'  # 设置默认的标签字体为加粗
plt.rcParams['axes.titleweight'] = 'bold'  # 设置默认的标题字体为加粗
# 定义绘图参数
font_size_ticks = 15
ticks_length = 10
parameter_size = 15
label_size = 25
legend_size = 16
title_size = 14
legendlength = 0.2
spines_width = 2
ticks_length_minor = 6
handletextpad = 0.5
ticks_width = 2.0
markersize2 = 10
data_key = 'sub12'  # 替换为实际的键名
savepath_mild = 'EDA_data/PH_net/delta/mild_pcc.mat'

# 加载MAT文件
data = loadmat(savepath_mild)
selected_data = data[data_key]
matrix1 = selected_data[1, :, :]

# 创建图形和轴
fig, ax1 = plt.subplots(figsize=(3.8, 3.4))  # 设置图形大小

# 绘制热图
im = ax1.imshow(matrix1, cmap='plasma', aspect='auto')  # 绘制热图

# 设置轴线宽度
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(spines_width)

# 设置纵轴刻度线属性
ax1.tick_params(axis='both', which='major', direction='out', width=ticks_width, labelsize=font_size_ticks)

# 移除横纵坐标的刻度和小短线
ax1.set_xticks([])  # 移除x轴上的刻度位置
ax1.set_yticks([])  # 移除y轴上的刻度位置

# 调整布局以适应所有标签
plt.tight_layout()

# 显示图表
plt.show()

# 如果你想保存图片，请取消下一行的注释
plt.savefig("eeg_net.png", dpi=300, bbox_inches='tight')
#%%
### plot scalpmap
import mne
import matplotlib.pyplot as plt

# 读取原始数据
raw = mne.io.read_raw_fif('data/severe/sub-01_task-motor-imagery_eeg.fif')

# 删除最后一个电极通道
raw.drop_channels([raw.ch_names[-1]])

# 设置电极排列为标准的10-20系统
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage, on_missing='ignore')  # 忽略缺失的电极

# 绘制脑电极分布
fig = raw.plot_sensors(show_names=True, kind='topomap', ch_type='eeg')

# 调整字体大小
for text in fig.axes[0].texts:  # fig.axes[0] 是绘图的主轴
    text.set_fontsize(20)  # 设置字体大小为12
# plt.savefig("eeg_scalpmap.png", dpi=300)
# plt.show()
#%%
#### PH
import numpy as np
from scipy.spatial import distance
import gudhi as gd
import pandas as pd
import matplotlib.pyplot as plt

# 定义绘图参数
font_size_ticks = 15
ticks_length = 10
parameter_size = 15
label_size = 20
legend_size = 16
title_size = 20
legendlength = 0.2
spines_width = 2
ticks_length_minor = 6
handletextpad = 0.5
ticks_width = 2
markersize2 = 10

# 设置全局字体样式和大小
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = font_size_ticks
plt.rcParams['ytick.labelsize'] = font_size_ticks
plt.rcParams['axes.titlesize'] = title_size
plt.rcParams['axes.labelsize'] = label_size

# 1. 读取 Excel 文件的特定工作表（假设工作表名称为 'Sheet1'）
data = pd.read_excel('EDA_data/PH_net/delta/mild_point_cloud/sub12_pointclouds.xlsx', sheet_name='17')

# 提取 x, y, z 坐标（假设这些列名是正确的）
points = data[['x', 'y', 'z']].values

# 2. 计算距离矩阵
dist_matrix = distance.pdist(points)
max_distance = np.max(dist_matrix)  # 计算最大距离

# 3. 构建单纯复形
rips_complex = gd.RipsComplex(points=points, max_edge_length=max_distance)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
persistence = simplex_tree.persistence()

# 使用 gudhi 绘制持续图
diagram_plot = gd.plot_persistence_diagram(persistence)

# 获取当前轴
ax = plt.gca()

# 调整刻度字体大小和宽度
ax.tick_params(axis='both', which='major', labelsize=font_size_ticks, width=ticks_width)

# 设置字体粗细
plt.setp(ax.get_xticklabels(), fontweight='bold')  # 设置x轴刻度标签的字体粗细
plt.setp(ax.get_yticklabels(), fontweight='bold')  # 设置y轴刻度标签的字体粗细

# 设置轴线宽度
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(spines_width)  # 边框粗细

# 设置标题和标签
ax.set_title('Persistence Diagram', fontsize=title_size, fontweight='bold')
ax.set_xlabel('Birth', fontsize=label_size, fontname='Times New Roman', fontweight='bold')
ax.set_ylabel('Death', fontsize=label_size, fontname='Times New Roman', fontweight='bold')

# 调整布局以适应所有标签
plt.tight_layout()

# 显示图表
plt.show()

# 如果你想保存图片，请取消下一行的注释
# plt.savefig("eeg_PH.png", dpi=300, bbox_inches='tight')

#%%
# score
import numpy as np
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体样式
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['font.serif'] = ['Times New Roman']

# 定义绘图参数
font_size_ticks = 22
ticks_length = 10
parameter_size = 15
label_size = 25
legend_size = 16
title_size = 14
legendlength = 0.2
spines_width = 2
ticks_length_minor = 6
handletextpad = 0.5
ticks_width = 2.0
markersize2 = 10
# 读取数据文件
file_path = r'participants.tsv'
df = pd.read_csv(file_path, sep='\t')

# 筛选 mild_case 和 severe_case 的患者数据
mild_case = ['03', '07','08','12','14','16','17','20','21','28','29','30','34','44','46','48','49','50']
severe_case = ['01','13','18','19','22','39','41','42','47']

# 提取患者编号的行（例如，sub-03 以 "03" 结尾）
df['Participant_ID'] = df['Participant_ID'].str[-2:]  # 提取编号后两位
mild_df = df[df['Participant_ID'].isin(mild_case)]
severe_df = df[df['Participant_ID'].isin(severe_case)]
labels = ['NIHSS', 'mRS', 'MBI']

# 绘制不同标签的对比图
for label in labels:
    plt.figure(figsize=(8, 5))
    plt.hist(mild_df[label], bins=10, alpha=0.5, label='Mild Case', color='blue')
    plt.hist(severe_df[label], bins=10, alpha=0.5, label='Moderate Case', color='red')
    # 使用ax对象来更精确地控制标签的字体样式
    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.set_xlabel(f'{label} Score', fontsize=font_size_ticks, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel('Frequency', fontsize=font_size_ticks, fontweight='bold', fontname='Times New Roman')


    # 调整刻度字体和边框线条
    plt.tick_params(axis='both', labelsize=legend_size, width=2)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(spines_width)
    plt.legend(loc='upper right', fontsize=legend_size)
    plt.tight_layout()
    plt.savefig(f"{label}_comparison.png", dpi=300)
    plt.show()
