# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:24:18 2024

@author: Zhanglu
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import Amplitude
from sklearn.preprocessing import StandardScaler
from scipy.signal import hilbert
# ========== 自定义：PLV 距离转换器 ==========
def plv_matrix(X):
    ch = X.shape[1]  # 通道数29 # (4000, 29)
    PLV = np.ones((ch, ch))  # 创建一个全1的矩阵 （channel，channe）
    for ch1 in range(ch - 1):  # 从0到28，不包括29
        for ch2 in range(ch1 + 1, ch):  # 不包括29
            # 获取两个通道的数据
            # 计算plv
            PLV[ch1, ch2] = phase_locking_value(X[:, ch1], X[:, ch2])
            PLV[ch2, ch1] = PLV[ch1, ch2]
    return PLV
# 计算plv
def phase_locking_value(theta1, theta2):
    # continuous hilbert phase function
    # 希尔伯特变换后，得到信号的瞬时相位值
    theta1 = hilbert(theta1)
    theta2 = hilbert(theta2)
    #  unwrap函数会将绝对值超过pi的角度增加或减少(2*k*pi)角度后换算回(-pi,pi)区间中
    theta1 = np.unwrap(np.angle(theta1))
    theta2 = np.unwrap(np.angle(theta2))
    complex_phase_diff = np.exp(complex(0, 1) * (theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff)) / len(theta1)
    return plv

class PLVDistance(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):  # X.shape = (n_samples, timesteps, channels)
        n_samples = X.shape[0]
        distance_matrices = []

        for sample in X:  # sample.shape = (timesteps, channels)
            plv = plv_matrix(sample)  # shape = (channels, channels)
            distance = 1 - plv
            distance_matrices.append(distance)

        return np.array(distance_matrices)

def compute_topological_features(sample, distance_method="pearson"):
    metric_list = [
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}}
     ]
    # metric_list 中的p是某个向量内部的距离；feature_union 中的order是整个特征与0参考线的距离。
    # feature_union 计算时会分别计算H0,H1同调特征，因此共有8个特征。
    # Amplitude 波幅s的持久性图。返回一个对象
    # PersistenceEntropy 持久性图的持久性熵，返回 Xt – 持久性熵：每个样本和每个同源维度一个值。沿轴 1 的索引 i 对应于 homology_dimensions_（同源维度在拟合中，按升序排序） 中的第i个同调维数。
    # make_union 的参数是转换器列表
    # feature_union = make_union(
    #      *[PersistenceEntropy(nan_fill_value=4)] +
    #      [Amplitude(**metric, order=2, n_jobs=-1) for metric in metric_list]
    # )
    feature_union = make_union(
        PersistenceEntropy(nan_fill_value=- 1.0),
        *[Amplitude(**metric, n_jobs=-1) for metric in metric_list]
    )
    # Creating the diagram generation pipeline  创建逻辑示意图生成管道
    # -1的意思是用所有cpu进行运算
    # 距离度量方式切换
    if distance_method == "pearson":
        distance_transformer = PearsonDissimilarity(n_jobs=-1)
    elif distance_method == "plv":
        distance_transformer = PLVDistance()
    else:
        raise ValueError("Unsupported distance method")

    diagram_steps = [[distance_transformer, VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1], n_jobs=-1)]]
    # make_pipeline从给定的估计器构造管道
    # make_union 连接多个转换器对象的结果
    # tda_union存的是一个列表管道，包括转换器名称和转换器里面的一些属性
    # make_pipeline 从给定的估计器构造管道。参数需要一个估计器列表
    tda_union = make_union(
        *[make_pipeline(*diagram_step) for diagram_step in diagram_steps],
        n_jobs=-1)
    # print(tda_union)
    # 现在 sample 的形状应该是 (4000, 29)，我们需要将其调整为 (1, 4000, 29)
    sample = sample[np.newaxis, :, :]
    X_pe = tda_union.fit_transform(sample)
    # 通过持续图，经过波的属性得到特征
    X_future = feature_union.fit_transform(X_pe)

    # normalize
    scaler = StandardScaler()
    topo_features = scaler.fit_transform(X_future.reshape(-1, 1))
    return topo_features
