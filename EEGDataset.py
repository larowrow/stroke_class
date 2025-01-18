# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import warnings
warnings.filterwarnings("ignore")
"""from torch.utils.data import Dataset, DataLoader"""

# 构建数据集
class EEGDataset(Dataset):
    # # X：特征矩阵(所有样本的), y：标签[0,1]，indices：测试对象的索引，sfreq：采样频率
    def __init__(self, feature, y, indices, root, transform=None):

        # 所有样本的特征
        self.feature = feature # （4000，31*15）（31*14）
        # 标签
        self.labels = y #（4000，1）
        # 训练或者测试样本窗口的索引
        self.indices = indices

        print("样本个数：")
        print(len(self.indices)) # 5566
        self.transform = transform

        # 完全连通无向图so：31*31边
        self.root = root # 脑网络的路径（所有样本）
        # plv、pcc、pli 作为邻接矩阵
        self.network = np.load(self.root, allow_pickle=True)

    # returns size of dataset = number of epochs
    def __len__(self):
        # indices：测试对象的索引
        return len(self.indices)

    # retrieve one sample from the dataset after applying all transforms
    # #应用所有转换后，从数据集中检索一个样本
    def __getitem__(self, idx):
        # torch.is_tensor()如果传递的对象是PyTorch张量，则方法返回True
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 将输入idx（self.indexes中的范围从0到_len__（））映射到整个数据集中的idx（self.epochs中
        # assert idx < len(self.indices)
        # 拿到测试样本的索引
        idx = self.indices[idx]
        # print("idx")
        # print(idx)

        # 拿到测试样本的节点特征
        node_features = self.feature[idx, :]
        # torch.from_numpy将数组转换为张量
        "------------------------------------------------------------"
        # node_features = torch.from_numpy(node_features.reshape(20, 15)) #（31，15）
        node_features = torch.from_numpy(node_features.reshape(29, 15)) #（31，15）
        # 单独拿出时域特征特征
        # node_features = torch.from_numpy(node_features.reshape(20, 15)[:,0:13]) #（31，13）
        # 单独拿出频域特征特征
        # node_features = torch.from_numpy(node_features.reshape(31, 15)[:,13:11]) #（31，1）[:,13:11]
        # 单独拿出脑网络特征
        # node_features = torch.from_numpy(node_features.reshape(31, 15)[:,11:15]) #（31，4）
        "------------------------------------------------------------"
        node_features = torch.tensor(node_features, dtype=torch.float32)
        # spectral coherence between 2 montage channels!
        # 两个通道之间的连通性（邻接矩阵）！[31 * 31]
        node_network = self.network[idx, :]
        node_network = torch.from_numpy(node_network.reshape(29, 29))
        node_network = torch.tensor(node_network, dtype=torch.float32)

        data = Data(x=node_features,
                    adj=node_network,
                    dataset_idx=idx,
                    y=self.labels[idx]
                    # pos=None, norm=None, face=None, **kwargs
                    )

        return data



