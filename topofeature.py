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

    feature_union = make_union(
        PersistenceEntropy(nan_fill_value=- 1.0),
        *[Amplitude(**metric, order=1, n_jobs=-1) for metric in metric_list]
    )

    diagram_steps = [
        [
            PearsonDissimilarity(n_jobs=-1),  # 通过皮尔逊不相关系数，得到距离矩阵
            VietorisRipsPersistence(n_jobs=-1), # 过滤距离矩阵，得到持续图
        ]
    ]

    tda_union = make_union(
        *[make_pipeline(*diagram_step) for diagram_step in diagram_steps],
        n_jobs=-1)

    sample = sample[np.newaxis, :, :]
    X_pe = tda_union.fit_transform(sample)
    # 通过持续图，经过波的属性得到特征
    X_future = feature_union.fit_transform(X_pe)

    # normalize
    scaler = StandardScaler()
    topo_features = scaler.fit_transform(X_future.reshape(-1, 1))
    return topo_features
