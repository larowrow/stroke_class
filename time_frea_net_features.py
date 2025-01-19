'''
计算时域和频域和网络特征
'''

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import warnings
import networkx as nx
from scipy.signal import butter, filtfilt, welch
warnings.filterwarnings("ignore")

# 时域特征
def getTimeFeatures(sample):

    # mean absolute value 平均绝对值，axis=0 按照列计算
    # [..., np.newaxis] 行变列
    mav = abs(sample).mean(axis=0)[..., np.newaxis]
    # # 标准差
    std = np.std(sample, axis=0)[..., np.newaxis]
    # # skewness 偏斜
    sample_skewness = skew(sample, axis=0)[..., np.newaxis]
    # # kurtosis 峰度
    sample_kurtosis = kurtosis(sample, axis=0)[..., np.newaxis]

    # np.hstack():在水平方向上平铺
    time_features = np.hstack((mav, std, sample_skewness, sample_kurtosis))
    # normalize
    scaler = StandardScaler()
    time_features = scaler.fit_transform(time_features)
    return time_features



##度、介数、聚类系数

def getNetFeatures(matrix):
    """
    根据邻接矩阵计算网络特征
    :param matrix: 邻接矩阵，形状为 (n_nodes, n_nodes)
    :return: 网络特征矩阵，形状为 (n_nodes, n_features)
    """
    # Step 1: 将邻接矩阵转换为 NetworkX 图
    G = nx.from_numpy_array(np.array(matrix))

    # 节点的介数中心性 (Betweenness Centrality)
    betweenness = np.array(list(nx.betweenness_centrality(G).values()))[..., np.newaxis]  # (n_nodes, 1)

    # 节点的聚类系数 (Clustering Coefficient)
    clustering = np.array(list(nx.clustering(G).values()))[..., np.newaxis]  # (n_nodes, 1)

    # Step 3: 构造特征矩阵
    net_features = np.hstack((betweenness, clustering))

    # Step 4: 特征归一化
    scaler = StandardScaler()
    net_features = scaler.fit_transform(net_features)

    return net_features


"""频域特征"""
def calculate_band_power(data, fs, nperseg=1024):
    """
    计算EEG数据的总功率
    :param data: EEG 数据的 NumPy 数组 (4000,)
    :param fs: 采样频率
    :param nperseg: Welch 方法中分段的长度，默认 1024
    :return: 数据的总功率
    """
    # Step 1: 使用 Welch 方法计算功率谱密度 (PSD)
    freqs, psd = welch(data, fs, nperseg=nperseg)  # 计算功率谱密度

    # Step 2: 计算信号的总功率
    total_power = np.sum(psd)  # 功率

    return total_power
    
def getFreFeatures(sample):
    # 频域特征
    totalnode = np.size(sample,1)
    freq_features = np.zeros((totalnode, 1))
    
    sfreq = 500
    # 循环每一个通道
    for channel in range(totalnode):
        freq_features[channel,0] = calculate_band_power(sample[:,channel], sfreq,  nperseg=1024)

    # normalize
    scaler = StandardScaler()
    freq_features = scaler.fit_transform(freq_features)
    
    return freq_features
    



