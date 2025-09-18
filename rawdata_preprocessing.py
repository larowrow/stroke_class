import os
import mne
import pandas as pd
from mne.preprocessing import ICA
import numpy as np

# 设置路径
data_path = 'E:/lin/stroke/stroke_GCN/edffile/edffile'  # 原始数据路径
data_save = 'E:/lin/stroke/stroke_GCN/edffile0'  # 处理后数据保存路径
electrodes_path = 'task-motor-imagery_electrodes.tsv'  # 电极位置文件路径

# 获取所有子目录（sub-01 到 sub-50）
subject_folders = [f'sub-{i:02d}' for i in range(1, 51)]  # 生成 'sub-01', 'sub-02', ..., 'sub-50'

# 确保保存目录存在，如果没有则创建
if not os.path.exists(data_save):
    os.makedirs(data_save)

# 读取电极位置文件
electrodes_df = pd.read_csv(electrodes_path, sep='\t')

# 构建电极位置字典
electrode_positions = {}
for index, row in electrodes_df.iterrows():
    label = row['name']
    x, y, z = row['X'], row['Y'], row['Z']
    electrode_positions[label] = np.array([x, y, z])

# 创建电极布局
montage = mne.channels.make_dig_montage(ch_pos=electrode_positions, coord_frame='head')

# 遍历所有子文件夹
for subject in subject_folders:
    # 构造当前子文件夹中 EEG 数据的路径
    eeg_file_path = os.path.join(data_path, subject, f'eeg/{subject}_task-motor-imagery_eeg.edf')

    if os.path.exists(eeg_file_path):  # 确保文件存在
        print(f'Processing {eeg_file_path}...')

        # 读取 EEG 数据
        raw = mne.io.read_raw_edf(eeg_file_path, preload=True)

        # 去掉 CPz 电极
        raw.drop_channels(['CPz'])

        # 设置电极位置
        raw.set_montage(montage, on_missing='warn')

        # 标记 EOG 通道 (如果需要)
        raw.set_channel_types({'HEOL': 'eog', 'HEOR': 'eog'})

        # 使用 ICA 进行去伪影分析
        ica = ICA(n_components=10, method='infomax', max_iter=800, random_state=97)
        ica.fit(raw)

        # 可视化 ICA 成分
        ica.plot_sources(raw)
        ica.plot_components()
        pick_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        raw_filtered = raw.copy().filter(l_freq=0, h_freq=40)
        ica.plot_properties(raw_filtered, picks=pick_id)
        # 提示用户输入眼动伪影成分
        eye_artifacts = input("Enter the ICA components for eye movement artifacts (comma separated): ")
        if eye_artifacts:
            # 解析用户输入的成分列表（以逗号分隔）
            eye_artifacts = [int(i.strip()) for i in eye_artifacts.split(',')]
            print(f"Excluding ICA components {eye_artifacts} based on user input.")
            # 排除指定的眼动伪影成分
            ica.exclude = eye_artifacts

            # 应用 ICA 去除伪影
            raw_reconstructed = ica.apply(raw)
        else:
            # 如果用户没有输入，默认不去除成分
            raw_reconstructed = raw

        # # 去除 EOG 电极
        # raw_reconstructed.drop_channels(['HEOL', 'HEOR'])
        #
        # # 去除最后一行数据
        # raw_reconstructed = raw_reconstructed.copy()  # 创建数据的副本以避免修改原始数据
        # raw_reconstructed.annotations.delete([len(raw_reconstructed.annotations)-1])
        #
        output_path = os.path.join(data_save, f'{subject}_task-motor-imagery_eeg_processed.fif')
        raw_reconstructed.save(output_path, overwrite=True)
        print(f'Processed data saved to {output_path}')

    else:
        print(f'File not found: {eeg_file_path}')