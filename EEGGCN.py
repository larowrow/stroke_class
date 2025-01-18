# 稠密图
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv,BatchNorm

# 原始模型
# 构建图卷积网络
class EEGGCN(torch.nn.Module):
    # 初始化
    def __init__(self):
        super(EEGGCN,self).__init__()
        # 设置CPU生成随机数的种子，方便下次复现实验结果。
        torch.manual_seed(42) # 42
        # 四个,指定输入和输出特征的维数
        self.width = 16
        self.conv1 = DenseGCNConv(15,self.width) # （15，16）
        self.conv2 = DenseGCNConv(self.width,self.width//4) #（16，4）

        # BN层
        self.batchnorm1 = BatchNorm(self.width, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm2 = BatchNorm(self.width//4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # fc
        self.linear1 = torch.nn.Linear(29*self.width//4,29)
        self.linear2 = torch.nn.Linear(29,2)

    def forward(self,f,adj,batch):

        output1 = self.conv1(f, adj) #f(batch,node_number,feature_number)   adj(batch,node_number,node_number); output1(batch,node_number,self.width)
        input21 = output1.view([-1, self.width]) #output1(batch,node_number,self.width); input21(batch*node_number,self.width)
        input21 = F.dropout(F.leaky_relu(self.batchnorm1(input21), negative_slope=0.01), p=0.2,
                      training=self.training) #input21(batch*node_number,self.width);input21(batch*node_number,self.width)

        input21 = input21.view([-1, 29, self.width])#input21(batch*node_number,self.width); input21(batch,node_number,self.width)


        output21 = self.conv2(input21,adj)#input21(batch,node_number,self.width); output21(batch,node_number,self.width//4)
        output = output21.view([-1, self.width//4])#output21(batch,node_number,self.width//4); output(batch*node_number,self.width//4)
        output = F.dropout(F.leaky_relu(self.batchnorm2(output), negative_slope=0.01), p=0.2,
                      training=self.training)#output(batch*node_number,self.width//4); output(batch*node_number,self.width//4)

        output = output.view([-1, 29, self.width//4])#output(batch*node_number,self.width//4); output(batch,node_number,self.width//4)
        # # view函数将张量变形成一维向量形式，总特征数不变，为全连接层做准备
        output = output.view(-1,29*self.width//4)#output(batch,node_number,self.width//4); output(batch,node_number*self.width//4)

        output = self.linear1(output)#output(batch,node_number*self.width//4);output(batch,node_number)
        output = F.dropout(F.leaky_relu(output, negative_slope=0.01), p=0.2, training=self.training)#output(batch,node_number); output(batch,node_number)

        output = self.linear2(output)#output(batch,node_number);output(batch,2)
        return F.log_softmax(output, dim=1)#output(batch,2) return(batch,2)



