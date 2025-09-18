import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm


class EEGCNN2(nn.Module):
    def __init__(self):
        super(EEGCNN2, self).__init__()
        # 设置随机数种子，方便复现实验结果
        torch.manual_seed(42)
        self.width = 8  # 第一个卷积层输出通道数为 8
        # 定义卷积层 1：5x5 卷积核，输出 8 通道
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=self.width, kernel_size=5, stride=3, padding=3)  # 卷积层

        # 池化层 1：2x2 最大池化
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dropout 层
        self.dropout1 = nn.Dropout(p=0.5)  # 添加 dropout，p=0.5 表示有 50% 的概率将神经元输出置为零

        # 批归一化层 1
        self.batchnorm1 = BatchNorm(self.width, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # 全连接层 1：256 神经元
        self.linear1 = nn.Linear(40, 256)  # 假设输入的 num_nodes = 29

        # 全连接层 2：输出 2 类别
        self.linear2 = nn.Linear(256, 2)

    def forward(self, f, adj, batch):
        # 假设输入的 f 形状是 (batch_size, num_nodes, num_features)
        # 转置输入特征矩阵，将其调整为 (batch_size, num_features, num_nodes) 适应 1D 卷积
        x = f.transpose(1, 2)  # (batch_size, num_features, num_nodes)

        # 卷积层 1 + 批归一化 + 激活函数
        x = F.relu(self.conv1(x))  # 使用 ReLU 激活
        x = self.pool1(x)  # 池化
        # 添加 Dropout 层
        x = self.dropout1(x)
        # 批量归一化
        x = self.batchnorm1(x)

        # 将数据展平，准备输入到全连接层
        x = x.view(x.size(0), -1)  # 展平为 (batch_size, num_nodes * num_features)

        # 全连接层 1
        x = F.relu(self.linear1(x))  # 使用 ReLU 激活 x的形状决定了self.linear1 = nn.Linear(64, 256) 中的64

        # 输出层（全连接层 2）
        x = self.linear2(x)  # 输出 2 个类别
        return F.log_softmax(x, dim=1)  # 对输出做 softmax
