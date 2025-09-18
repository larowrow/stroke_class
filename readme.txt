plotEEG_net_PH.py: figure 1画图：脑电，分数
Classification judgment.py：轻中症病人划分
eegdata_check.py: 查看eeg data
rawdata_preprocessing.py: 对公共数据（预处理过的）再次手动预处理
statistics_analysis.py: 病人基线资料分析
showdata.py: 用Data_preprocessing.py得到的结果计算PCC下H0,H1,H2个数
showdata_plv.py: 用Data_preprocessing.py得到的结果 计算PLV下H0,H1,H2个数
Data_preprocessing.py: 划分样本，创建重采样好的特征矩阵
topofeature_plv.py: Data_preprocessing.py调用的计算高阶特征的函数
topofeature.py: Data_preprocessing.py调用的计算高阶特征的函数
network.py:网络特征计算
time_frea_net_features.py: 时频域特征
plot_H012_bar.py 画图显示H0H1H2
feature_select_MI_allband_allnet.py: 两种指标下计算重要特征
EEGDataset.py: 封装数据以便后续传入GCN
EEGGCN1: GCN模型
loss_curve.py: loss曲线
plot_ROC.py: ROC曲线
After_MI_feature.py: 重要特征矩阵
loss_curve CNN2.py: loss曲线CNN2的
EEGCNN2.py: CNN2模型
test_CNN2.py: CNN2测试
train_CNN2.py: CNN2训练
test_SMOTE.py: GCN1测试
train_SMOTE.py: GCN1训练
random_Forest.py: 随机森林
After_MI_feature_local_global.py: 提取global, local特征矩阵
train_SMOTE_local_global.py: 训练global, local
test_SMOTE_local_global.py: 测试global, local
cycle_ratio.py: 计算圈比
boxplot_cycle.py：画圈数差
plot_Scalpmap_ver2.py: 统一热量脑地形图
MIDR_severity.py: 运动想象解码率与严重程度的相关性
mergeLI: 制作加入Laterality Index的特征矩阵
SHAP2: SHAP最终版
Grad_CAM：看电极（脑通道）的重要性
ERD_ERS: 从fif格式的EEG数据中提取运动想象任务的ERD/ERS特征；计算大脑偏侧化指数（LI）；使用SMOTE处理类别不平衡问题；保存处理后的特征用于后续分析
ERD:好像没用到
loss_curve_tiaocan: 画各种参数下的loss curve


