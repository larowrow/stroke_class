# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:33:57 2024

@author: Zhanglu
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
from matplotlib import rcParams
# 设置文件夹路径
data_path = 'CycleRatio_data'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 初始化一个字典来存储每一行的第2列数据
mild_data_lines = []
severe_data_lines = []

# 遍历指定路径中的所有txt文件
for filename in os.listdir(data_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_path, filename)
        
        # 读取txt文件，假设数据是以空格或制表符分隔的
        data = np.loadtxt(file_path)
        
        # 获取文件的行数（假设所有txt文件的行数相同）
        num_lines = data.shape[0]
        # 对每行的第2列进行分类存储
        for i in range(num_lines):
            second_column_value = data[i, 1]  # 第2列是索引1
            
            if 'mild' in filename:
                if i >= len(mild_data_lines):
                    mild_data_lines.append([])  # 如果是新的行，初始化一个列表
                mild_data_lines[i].append(second_column_value)
                
            elif 'severe' in filename:
                if i >= len(severe_data_lines):
                    severe_data_lines.append([])  # 如果是新的行，初始化一个列表
                severe_data_lines[i].append(second_column_value)
               

# 计算每行的差异：'mild' - 'severe'
diff_data = []
for i in range(len(mild_data_lines)):
    mild_values = mild_data_lines[i]
    severe_values = severe_data_lines[i]
    
    # 计算每行差异
    if len(mild_values) == len(severe_values):
        diff_values = np.array(mild_values) - np.array(severe_values)
        diff_data.append(diff_values)

# 使用颜色映射绘制每行的差异：横轴是行号，纵轴是 'mild' - 'severe' 的差异
color_map = cm.plasma(np.linspace(0, 1, 5))  # 使用viridis色标为5个点分配颜色

# 定义频段对应名称
frequency_bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']

for i in range(len(diff_data)):
    # 为每行绘制一个点线图
    x = [i + 1] * 5  # 横轴为行号，纵轴为每行的差异值
    y = diff_data[i]  # 每行的差异值

    # 'o-' 表示画点并连接线
    plt.plot(x, y, 'o-', color=color_map[0], markersize=8)  # 确保连接每个点

    # 绘制每个频段的点，并且连接
    for j in range(5):
        plt.plot(i + 1, diff_data[i][j], 'o-', color=color_map[j], label=frequency_bands[j], markersize=8)

# 设置图形标签
plt.xlabel('Rank Number', fontsize=20, fontname='Times New Roman', fontweight='bold')  # 调大坐标轴标签字体
plt.ylabel('Diff. of Cycle No. (Mild - Moderate)', fontsize=20, fontname='Times New Roman', fontweight='bold')  # 调大坐标轴标签字体

# 设置坐标轴刻度标签的字体大小和字体样式（Times New Roman）
plt.xticks(fontsize=14, fontname='Times New Roman', fontweight='bold')  # 设置x轴刻度
plt.yticks(fontsize=14, fontname='Times New Roman', fontweight='bold')  # 设置y轴刻度

# 设置图例：每个颜色对应一个频段名称，并调大图例的字体
handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[j], markersize=8, label=frequency_bands[j]) for j in range(5)]
plt.legend(handles=handles, loc='best', bbox_to_anchor=(1, 1), fontsize=16)

# 加粗边框
ax = plt.gca()  # 获取当前的轴对象
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
# 显示图形
plt.show()
plt.savefig("cycle_diff.png", dpi=300)