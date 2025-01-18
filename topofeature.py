# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:24:18 2024

@author: Zhanglu
"""
import numpy as np
from gtda.time_series import PearsonDissimilarity
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.pipeline import make_pipeline, make_union
from gtda.diagrams import Amplitude
from sklearn.preprocessing import StandardScaler
def compute_topological_features(sample):
    metric_list = [
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 1, "n_bins": 100}},
        {"metric": "landscape", "metric_params": {"p": 1, "n_layers": 2, "n_bins": 100}},
        {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}}
     ]
    # Amplitude 波幅s的持久性图。返回一个对象
    # PersistenceEntropy 持久性图的持久性熵，返回 Xt – 持久性熵：每个样本和每个同源维度一个值。沿轴 1 的索引 i 对应于 homology_dimensions_（同源维度在拟合中，按升序排序） 中的第i个同调维数。
    # make_union 的参数是转换器列表
    # feature_union = make_union(
    #      *[PersistenceEntropy(nan_fill_value=4)] +
    #      [Amplitude(**metric, order=2, n_jobs=-1) for metric in metric_list]
    # )
    feature_union = make_union(
        PersistenceEntropy(nan_fill_value=- 1.0),
        *[Amplitude(**metric, order=1, n_jobs=-1) for metric in metric_list]
    )
    # Creating the diagram generation pipeline  创建逻辑示意图生成管道
    # -1的意思是用所有cpu进行运算
    diagram_steps = [
        [
            PearsonDissimilarity(n_jobs=-1),  # 通过皮尔逊不相关系数，得到距离矩阵
            VietorisRipsPersistence(n_jobs=-1), # 过滤距离矩阵，得到持续图
        ]
    ]
    # make_pipeline从给定的估计器构造管道
    # make_union 连接多个转换器对象的结果
    # tda_union存的是一个列表管道，包括转换器名称和转换器里面的一些属性
    # make_pipeline 从给定的估计器构造管道。参数需要一个估计器列表
    tda_union = make_union(
        *[make_pipeline(*diagram_step) for diagram_step in diagram_steps],
        n_jobs=-1)
    # print(tda_union)
    # 现在 sample 的形状应该是 (4000, 31)，我们需要将其调整为 (1, 4000, 31)
    sample = sample[np.newaxis, :, :]
    X_pe = tda_union.fit_transform(sample)
    # 通过持续图，经过波的属性得到特征
    X_future = feature_union.fit_transform(X_pe)

    # normalize
    scaler = StandardScaler()
    topo_features = scaler.fit_transform(X_future.reshape(-1, 1))
    return topo_features
