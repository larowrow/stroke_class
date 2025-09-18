# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:04:24 2024

@author: Zhanglu
"""
# import numpy as np
import os
import pandas as pd
from glob2 import glob
import mne
from time_frea_net_features import *
from network import *
from topofeature_plv import *
import warnings
from imblearn.over_sampling import SMOTE
from mne import filter
warnings.filterwarnings("ignore")


# 创建数据集索引函数
def process_patient_group(group_label):
    """
    处理指定组别的患者（severe 或 mild），提取其 EEG 数据窗口元数据
    """
    set_file_list = glob(os.path.join(rootPath, group_label, '*.fif'))

    unique_patient_ids = list(np.unique([x.split("\\")[-1].split("sub-")[1].split("_")[0] for x in set_file_list]))
    print(f"Unique {group_label} IDs: {len(unique_patient_ids)}")

    for item in unique_patient_ids:
        file_handler.write(f"{item}\n")

    for idx, patient_id in enumerate(unique_patient_ids):
        print(f"{patient_id} : {idx + 1}/{len(unique_patient_ids)}")

        patient_set_file_list = glob(os.path.join(rootPath, group_label, f"sub-{patient_id}_*.fif"))
        assert len(patient_set_file_list) >= 1

        labels = [x.split("\\")[-2] for x in patient_set_file_list]
        assert labels == [labels[0]] * len(labels)

        label = labels[0]
        label_count[label] += 1

        # 生成窗口元数据
        timess = 160
        for start_sample_index in range(0, int(timess * SAMPLING_FREQ), WINDOW_LENGTH_SAMPLES):
            end_sample_index = start_sample_index + WINDOW_LENGTH_SAMPLES
            if end_sample_index > int(timess * SAMPLING_FREQ):
                break

            row = {
                "patient_id": patient_id,
                "raw_file_path": patient_set_file_list[0],
                "sampling_freq": SAMPLING_FREQ,
                "start_sample_index": start_sample_index,
                "end_sample_index": end_sample_index,
                "text_label": label,
                "numeric_label": 0 if group_label == "mild" else 1  # mild 为 0，severe 为 1
            }
            dataset_index_rows.append(row)

# 设置参数
bands = {
    "Delta": [1, 3],
    "Theta": [4, 7],
    "Alpha": [8, 13],
    "Beta": [14, 30],
    "Gamma": [31, 40]
}
rootPath = "data"
time_three = 8.0
SAMPLING_FREQ = 500.0
WINDOW_LENGTH_SAMPLES = int(time_three * SAMPLING_FREQ)

for band_name, band_range in bands.items():
    resultpath = os.path.join(rootPath, f"time_{int(time_three)}", band_name)
    
    if not os.path.exists(resultpath):
        os.makedirs(resultpath)

    # 初始化
    dataset_index_rows = []
    label_count = {"severe": 0, "mild": 0}

    # 打开文件，写入患者 ID 信息
    with open(os.path.join(resultpath, 'subjects.txt'), 'w') as file_handler:
      
        # 处理 mild 组
        process_patient_group("mild")
        
        # 处理 severe 组
        process_patient_group("severe")
        
    # 创建 DataFrame 并保存到 CSV
    df = pd.DataFrame(dataset_index_rows)
    df.to_csv(os.path.join(resultpath, "stroke_window_index.csv"), index=False)
    print(label_count)
    
    
    # 特征和网络矩阵的初始化
    totalnode = 29
    pccfeature_array = np.zeros((len(dataset_index_rows), totalnode * 15))
    plvfeature_array = np.zeros((len(dataset_index_rows), totalnode * 15))

    
    plvbinary_array = np.zeros((len(dataset_index_rows), totalnode * totalnode))
    pccbinary_array = np.zeros((len(dataset_index_rows), totalnode * totalnode))

    
    # 按文件路径分组
    grouped_df = df.groupby("raw_file_path")
    print(f"Total Windows: {len(df)}")
    for raw_file_path, group_df in grouped_df:
        print(f"Processing File: {raw_file_path}")
    
        raw = mne.io.read_raw_fif(raw_file_path, preload=True)
        raw_data = raw.get_data(picks=np.arange(29))[:, :raw.n_times // 2]
        filtered_data = filter.filter_data(raw_data, sfreq=raw.info['sfreq'], l_freq=band_range[0], h_freq=band_range[1])
        for window_idx in group_df.index.tolist():
            start_sample = group_df.loc[window_idx]['start_sample_index']
            stop_sample = group_df.loc[window_idx]['end_sample_index']
    
            window_data = filtered_data[:, start_sample:stop_sample] #(29,4000)
            sample = window_data.T #(4000,29)
            # 计算脑网络
            PLV = plv_matrix(sample)
            PLVb = weight2binary(PLV)
            plvbinary_array[window_idx, :] = PLVb.flatten()
            
            PCC = pcc_matrix(sample)
            PCCb = weight2binary(PCC)
            pccbinary_array[window_idx, :] = PCCb.flatten()

    
            # 提取时频域网络特征
            time_features = getTimeFeatures(sample)
            freq_features = getFreFeatures(sample)
            # 提取拓扑特征
            topo = compute_topological_features(sample, distance_method="plv")#, distance_method="pearson"
            topo_plv = np.tile(topo.T, (29, 1))
            topo = compute_topological_features(sample)  # , distance_method="pearson"
            topo_pcc = np.tile(topo.T, (29, 1))
            pcc_features = getNetFeatures(PCCb)
            plv_features = getNetFeatures(PLVb)

            time_fre_pcc_features = np.hstack((time_features, freq_features, pcc_features, topo_pcc))
            time_fre_plv_features = np.hstack((time_features, freq_features, plv_features, topo_plv))

            
            features = time_fre_pcc_features.flatten()
            pccfeature_array[window_idx, :] = features
            
            features = time_fre_plv_features.flatten()
            plvfeature_array[window_idx, :] = features


            print("[WINDOW] FEATURES DONE!")
    
    
    # 重采样
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # 应用SMOTE进行数据增强
    y = df["numeric_label"].to_numpy()
    pccfeature_array_resampled, y_resampled = smote.fit_resample(pccfeature_array, y)
    plvfeature_array_resampled, y_resampled = smote.fit_resample(plvfeature_array, y)

    
    plvbinary_array_resampled, y_resampled = smote.fit_resample(plvbinary_array, y)
    # 将重采样后的数组转换为0或1
    plvbinary_array_resampled = np.clip(plvbinary_array_resampled, 0, 1).astype(int)


    pccbinary_array_resampled, y_resampled = smote.fit_resample(pccbinary_array, y)
    # 将重采样后的数组转换为0或1
    pccbinary_array_resampled = np.clip(pccbinary_array_resampled, 0, 1).astype(int)
    # 输出结果
    print("原始样本数量:", np.bincount(y))
    print("增强后的样本数量:", np.bincount(y_resampled))
    
    # # 保存结果：[1节点的所有特征,2节点的所有特征,...,31节点的所有特征]
    np.save(os.path.join(resultpath, "pcc_net_time_freq.npy"), pccfeature_array_resampled)
    np.save(os.path.join(resultpath, "plv_net_time_freq.npy"), plvfeature_array_resampled)
    # np.save(os.path.join(resultpath, "pli_net_time_freq.npy"), plifeature_array_resampled)
    np.save(os.path.join(resultpath, "plv_network.npy"), plvbinary_array_resampled)
    np.save(os.path.join(resultpath, "pcc_network.npy"), pccbinary_array_resampled)
    # np.save(os.path.join(resultpath, "pli_network.npy"), plibinary_array_resampled)
    np.save(os.path.join(resultpath, "y_label.npy"), y_resampled)
    print("ALL ARRAYS SAVED TO DISK!")
