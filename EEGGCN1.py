import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, BatchNorm


class EEGGCN1(torch.nn.Module):
    # 初始化
    def __init__(self):
        super(EEGGCN1, self).__init__()
        # 设置CPU生成随机数的种子，方便下次复现实验结果。
        torch.manual_seed(42)  # 42
        # 四个,指定输入和输出特征的维数
        self.width = 14
        self.conv1 = DenseGCNConv(12, self.width)  # （feature，self.width）

        # BN层
        self.batchnorm1 = BatchNorm(self.width, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # fc层
        self.linear1 = torch.nn.Linear(29 * self.width, 29)  # 修改为适应单卷积层的输出
        self.linear2 = torch.nn.Linear(29, 2)

    def forward(self, f, adj, batch):
        # 单个卷积层
        output1 = self.conv1(f,
                             adj)  # f(batch,node_number,feature_number)   adj(batch,node_number,node_number); output1(batch,node_number,self.width)

        # 展平操作
        output1 = output1.view(
            [-1, self.width])  # output1(batch,node_number,self.width); input1(batch*node_number,self.width)
        output1 = F.dropout(F.leaky_relu(self.batchnorm1(output1), negative_slope=0.01), p=0.7,
                            training=self.training)  # output1(batch*node_number,self.width)

        # 重新调整形状
        output1 = output1.view(
            [-1, 29, self.width])  # output1(batch*node_number,self.width); output1(batch,node_number,self.width)

        # 展平
        output1 = output1.view(-1,
                               29 * self.width)  # output1(batch,node_number,self.width); output1(batch,node_number*self.width)

        # 全连接层1
        output1 = self.linear1(output1)  # output1(batch,node_number*self.width); output1(batch,node_number)

        # 激活函数 + Dropout
        output1 = F.dropout(F.leaky_relu(output1, negative_slope=0.01), p=0.7,
                            training=self.training)  # output1(batch,node_number); output1(batch,node_number)

        # 全连接层2
        output1 = self.linear2(output1)  # output1(batch,node_number); output1(batch,2)

        # 输出softmax
        return F.log_softmax(output1, dim=1)  # output1(batch,2) return(batch,2)
