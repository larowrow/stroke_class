from sklearn.model_selection import KFold
import torch
from EEGGCN1 import EEGGCN1
from torch.optim import lr_scheduler
import time
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.data import DenseDataLoader
from EEGDataset import EEGDataset
from torchvision.transforms import Compose, ToTensor
import warnings
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
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
        X_batch = batch #X_batch: DataBatch(dataset_idx=[32], adj=[32, 29, 29], x=[32, 29, 12], y=[32])
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


def test(fold, epoch, testloader=None, training=False):

    global best_acc, best_epoch, epochs_no_improve, patience, \
        early_stop, test_losses, batch_size, bestModel
    model.eval().to(device)
    test_loss = 0
    accuracy = 0
    y_preds = []
    y_true = []

    bestModel = os.path.join(resultpath, "model_local", f"{connetc}_kf_{fold}_local.pth")


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

            # if not training:
            #     # append用于向列表末尾追加指定元素
            #     # 所有样本预测值（0，1）
            #     y_preds.append(torch.argmax(pred, dim=1).to(device))
            #     # 所有样本的真实值（0，1）
            #     y_true.append(label.to(device))
            if not training:
                # 计算 AUC 使用 softmax 概率值
                y_prob = torch.softmax(pred, dim=1)[:, 1]  # 取正类概率
                y_preds.append(y_prob.cpu())  # 概率值而非类别
                y_true.append(label.cpu())

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
    connetc = "plv"

    resultpath = rootPath + "/" + time_w  + "/" + band + "/"
    # 29*12的特征
    feature = np.load(resultpath +f'MI_{connetc}_localfeature.npy')
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
    # 初始化保存结果的数组
    all_train_losses = []
    all_train_accs = []
    all_valid_losses = []
    all_valid_accs = []
    all_aucs = []
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
        # 记住：每一折都要实例化新的模型，不然模型会学到测试集的东西
        # 实现卷积类
        model = EEGGCN1()
        model = model.to(device)
        # 参数设置
        lr = 0.005
        # early stopping variables
        # 早停止变量,全局变量
        epochs_no_improve = 0
        patience = 150
        early_stop = False
        best_acc = 0
        best_epoch = 0
        epochs = 150
        lr_step = 50
        weight_decay = 1e-4
        dropoutp=0.6
        # cross entropy 交叉熵
        criterion = nn.CrossEntropyLoss()
        # 优化器,weight decay（权值衰减）使用的目的是防止过拟合
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # 调整学习率
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

        print("Starting training...")
        # 训练数据集和验证数据集的正确率和损失函数
        fold_train_losses, fold_train_accs = [], []
        fold_valid_losses, fold_valid_accs = [], []
        fold_aucs = []  # 存储每一折的AUC
        # 循环每一个epoch，就是所有训练样本都过一遍
        start = time.time()
        for epoch in range(1, epochs):
            train_loss, train_acc = train()
            valid_loss, valid_acc, y_preds, y_true = test(fold, epoch, val_dataloader)
            # Flatten the lists of tensors into a single 1D array for both y_preds and y_true
            y_preds = torch.cat([pred.cpu().view(-1) for pred in y_preds], dim=0).numpy()
            y_true = torch.cat([true.cpu().view(-1) for true in y_true], dim=0).numpy()

            auc = roc_auc_score(y_true, y_preds)
            fpr, tpr, _ = roc_curve(y_true, y_preds)
            if epoch % 10 == 0:
                print('\nepoch#: {} | train loss: {:.3f} | val loss: {:.3f} | train acc: {:.3f} | val acc: {:.3f}' \
                      .format(epoch, train_loss, valid_loss, train_acc * 100, valid_acc * 100))
                print('Best Acc: {:.3f} | Best Epoch: {} | epochs_no_improve: {}' \
                      .format(best_acc, best_epoch, epochs_no_improve))
                print(f"Device = {device}; Time per iter: {(time.time() - start):.3f} seconds")
            # 保存每一折的损失和准确率
            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_valid_losses.append(valid_loss)
            fold_valid_accs.append(valid_acc)
            fold_aucs.append(auc)
            if early_stop:
                print('Stopped')
                break  # 跳出循环
            scheduler.step()
    #     # 在每折训练结束后，将当前折的结果保存到全局数组中
    #     all_train_losses.append(fold_train_losses)
    #     all_train_accs.append(fold_train_accs)
    #     all_valid_losses.append(fold_valid_losses)
    #     all_valid_accs.append(fold_valid_accs)
    #     all_aucs.append(fold_aucs)
    # # 保存所有折的结果到文件中
    # os.makedirs(f"data/time_8/{band}/1loss_curve", exist_ok=True)
    # np.save(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_losses.npy", all_train_losses)  # shape: (10, epochs)
    # np.save(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_train_accs.npy", all_train_accs)  # shape: (10, epochs)
    # np.save(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_losses.npy", all_valid_losses)  # shape: (10, epochs)
    # np.save(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_valid_accs.npy", all_valid_accs)  # shape: (10, epochs)
    # np.save(f"data/time_8/{band}/1loss_curve/{connetc}_{lr}_{lr_step}_{dropoutp}_aucs.npy", all_aucs)
