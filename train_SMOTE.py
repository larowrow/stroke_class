from sklearn.model_selection import KFold
import torch
from EEGCNN import EEGCNN
from torch.optim import lr_scheduler
import pandas as pd
import time
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import DenseDataLoader
from EEGDataset import EEGDataset
from torchvision.transforms import Compose, ToTensor
import warnings
import os
warnings.filterwarnings("ignore")

"""十折交叉验证"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def train():
    model.train().to(device)
    running_loss = 0
    accuracy = 0

    for i,batch in enumerate(train_dataloader):#DataBatch(adj=[batch, node_number, node_number], x=[batch, node_number, feature_number], y=[batch], dataset_idx=[20])
        batch = batch.to(device, non_blocking=True)
        X_batch = batch
        # 标签
        label = torch.tensor(batch.y)

        optimizer.zero_grad()
        pred = model(f=X_batch.x, adj=X_batch.adj, batch=X_batch.batch)#X_batch.x(batch,node_number,feature_number) adj(batch,node_number,node_number)
        """pred = model(feature, network)"""
        # 计算损失函数和梯度
        loss = criterion(pred, label.to(device))#pred(20,2)
        loss.backward()
        running_loss += loss.item()
        # 更新参数
        optimizer.step()
        # 返回每一行中最大值的索引
        accuracy += (torch.argmax(pred, dim=1) == label).type(torch.float).mean().item()

    # 返回损失函数和正确率
    return running_loss/len(train_dataloader), accuracy/len(train_dataloader)


def test(fold, epoch, testloader=None, training=True):

    global best_acc, best_epoch, epochs_no_improve, patience, \
        early_stop, test_losses, batch_size, bestModel
    model.eval().to(device)
    test_loss = 0
    accuracy = 0
    y_preds = []
    y_true = []

    bestModel = os.path.join(resultpath, "model_CNN", f"{connetc}_kf_{fold}.pth")


    # 以下运算不进行反向求导，梯度计算，即参数不发生变化
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            batch = batch.to(device, non_blocking=True)
            X_batch = batch
            label = torch.tensor(batch.y)

            # 调用模型，得到测试结果
            pred = model(X_batch.x, X_batch.adj, X_batch.batch) #X_batch = (DataBatch) DataBatch(adj=[32, 31, 31], dataset_idx=[32], y=[32], x=[32, 31, 22])
            # 计算损失函数
            loss = criterion(pred, label.to(device))
            test_loss += loss.item()
            # (torch.argmax(pred, dim=1)返回每一行中最大值的索引（0，1）
            accuracy += (torch.argmax(pred, dim=1) == label.to(device)).type(torch.float).mean().item()

            if not training:
                # append用于向列表末尾追加指定元素
                # 所有样本预测值（0，1）
                y_preds.append(torch.argmax(pred, dim=1).to(device))
                # 所有样本的真实值（0，1）
                y_true.append(label.to(device))

    # 正确率
    clean_acc = 100. * accuracy / len(testloader)
    # print(clean_acc)
    if epoch == 1:
        best_acc = clean_acc
        "-----------------------------------------------------------------------------------"
    if (clean_acc >= best_acc and epoch > 100):
        # print('Saving..') # 变好则保存
        # 模型的存储
        torch.save(model.state_dict(), bestModel)
        epochs_no_improve = 0
        best_acc = clean_acc
        best_epoch = epoch
    else:
        epochs_no_improve += 1

    if epoch > 5 and epochs_no_improve == patience:
        print('Early stopping!')
        early_stop = True

    if not training:
        return test_loss / len(testloader), accuracy / len(testloader), y_preds, y_true
    else:
        return test_loss / len(testloader), accuracy / len(testloader)


if __name__ == "__main__":
    rootPath = "data"
    time_w = "time_8"
    band = 'Gamma'
    connetc = "pli"

    resultpath = rootPath + "/" + time_w  + "/" + band + "/"
    # 31*15的特征
    feature = np.load(resultpath +f'{connetc}_net_time_freq.npy')
    # 脑网络-邻接矩阵
    file_path = resultpath + connetc + '_network.npy'
    y = np.load(resultpath + 'y_label.npy',allow_pickle=True)

    SEED = 42
    np.random.seed(SEED) # numpy形式下随机种子，数组
    torch.manual_seed(SEED) # pytorch形式下的随机种子，张量

    num_samples_per_class = 360
    total_samples = 2 * num_samples_per_class

    # 创建训练和测试划分
    train_indices, test_indices = train_test_split(np.arange(total_samples), test_size=0.20, random_state=SEED,
                                                   stratify=y)# len(train_indices)=576; len(test_indices)=144

    # 输出测试样本的索引
    with open(resultpath + f'test_indices_{connetc}.txt', 'w') as file_handler:
        for index in test_indices:
            file_handler.write("{}\n".format(index))

    # K折交叉验证
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 开始K折交叉验证
    for fold, (train_id_idx, val_id_idx) in enumerate(skf.split(train_indices)):
        print('**' * 10, '第', fold + 1, '折', 'ing....', '**' * 10)

        # 加载训练和验证数据集
        # feature：特征矩阵, y：标签，test_indices
        train_dataset = EEGDataset(feature=feature, y=y, indices=train_id_idx, root=file_path,
                                   transform=Compose([ToTensor()]))#train_dataset(518)
        # train_dataset, val_dataset = train_test_split(train_dataset, test_size = 0.30,random_state=SEED)
        val_dataset = EEGDataset(feature=feature, y=y, indices=val_id_idx, root=file_path,
                                 transform=Compose([ToTensor()]))#val_dataset(58)
        # 数据集批处理
        train_dataloader = DenseDataLoader(dataset=train_dataset, batch_size=32,
                                           shuffle=True, num_workers=0)
        val_dataloader = DenseDataLoader(dataset=val_dataset, batch_size=32,
                                         shuffle=False, num_workers=0)
        # 实现卷积类
        model = EEGCNN()
        model = model.to(device)
        # 参数设置
        lr = 0.001
        # early stopping variables
        # 早停止变量,全局变量
        epochs_no_improve = 0
        patience = 200
        early_stop = False
        best_acc = 0
        best_epoch = 0
        epochs = 200
        lr_step = 70

        # cross entropy 交叉熵
        criterion = nn.CrossEntropyLoss()
        # 优化器,weight decay（权值衰减）使用的目的是防止过拟合
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # 调整学习率
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

        print("Starting training...")
        # 训练数据集和验证数据集的正确率和损失函数
        train_losses, train_accs = [], []
        valid_losses, valid_accs = [], []
        # 循环每一个epoch，就是所有训练样本都过一遍
        start = time.time()
        for epoch in range(1, epochs):
            train_loss, train_acc = train()
            valid_loss, valid_acc = test(fold, epoch, val_dataloader)
            if epoch % 10 == 0:
                print('\nepoch#: {} | train loss: {:.3f} | val loss: {:.3f} | train acc: {:.3f} | val acc: {:.3f}' \
                      .format(epoch, train_loss, valid_loss, train_acc * 100, valid_acc * 100))
                print('Best Acc: {:.3f} | Best Epoch: {} | epochs_no_improve: {}' \
                      .format(best_acc, best_epoch, epochs_no_improve))
                print(f"Device = {device}; Time per iter: {(time.time() - start):.3f} seconds")

            if early_stop:
                print('Stopped')
                break  # 跳出循环
            scheduler.step()

