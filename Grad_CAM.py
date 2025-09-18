import torch
from EEGGCN import EEGGCN
import pandas as pd
import numpy as np
from torch_geometric.data import DenseDataLoader
from EEGDataset import EEGDataset
from torchvision.transforms import Compose, ToTensor
import os
import warnings
warnings.filterwarnings("ignore")

# 获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 获取特征图的函数
def farward_hook(module, input, output):
    feaure_block.append(output)

# 计算gard_CAM
def gard_cam(feature_map, grads): # (29,4)
    cam = np.zeros(feature_map.shape[0], dtype=np.float32)  # (29,)
    # 梯度图中，每个通道计算均值得到一个值，作为对应特征图通道的权重
    weights = np.mean(grads, axis=0) # 4
    for i, w in enumerate(weights):
        cam += w * feature_map[:,i]	# 特征图加权和 (29,)
    cam = np.maximum(cam, 0) # relu
    # cam = cam / cam.max()
    return cam

if __name__=='__main__':
    rootPath = "data"
    t = "time_8"
    band = 'Gamma'
    netc = "pli"
    resultpath = rootPath + "/" + t + "/" + band + "/"

    feature = np.load(resultpath +f'{netc}_net_time_freq.npy')
    # 读取所有时间窗样本的标签
    y = np.load(resultpath + 'y_label.npy', allow_pickle=True)
    # 权重矩阵
    file_path = resultpath + netc + '_network.npy'
    bestModel = os.path.join(resultpath, "model", f"{netc}_kf_4.pth")

    SZ_DATASET_INDEX = pd.read_csv(resultpath + 'stroke_window_index.csv')

    """排序之后，训练、测试、验证是相同的窗口"""
    subjects = np.unique(SZ_DATASET_INDEX["patient_id"]).astype("str")

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    subjects_indices = []
    for id in subjects:
        # id为str，astype强制类型转换
        si = SZ_DATASET_INDEX.index[SZ_DATASET_INDEX["patient_id"].astype("str") == id].tolist()
        subjects_indices.extend(si)

    # In[]
    test_dataset = EEGDataset(feature=feature, y=y, indices=subjects_indices, root=file_path,
                              transform=Compose([ToTensor()]))
    test_dataloader = DenseDataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # forward
    # 在前向推理时，会生成特征图和预测值
    array = np.zeros((29,len(test_dataset)))
    for i, batch in enumerate(test_dataloader):

        model = EEGGCN()
        # 加载模型
        state_dict = torch.load(bestModel)
        print("最好模型路径：")
        print(bestModel)
        model.load_state_dict(state_dict)
        model.eval()

        grad_block = []  # 存放grad图
        feaure_block = []  # 存放特征图

        model.conv2.register_forward_hook(farward_hook)
        model.conv2.register_backward_hook(backward_hook)

        X_batch = batch
        label = torch.tensor(batch.y)

        pred = model(X_batch.x, X_batch.adj, X_batch.batch)
        # 计算损失函数
        max_idx = np.argmax(pred.cpu().data.numpy(),1)
        print("predict:{}".format(max_idx))
        # if max_idx == label:
        # backward
        model.zero_grad()
        # 取最大类别的值作为loss，这样计算的结果是模型对该类最感兴趣的cam图
        print(pred)
        class_loss = pred[0, max_idx]
        class_loss.backward()  # 反向梯度，得到梯度图

        # grads
        grads_val = grad_block[0].cpu().data.numpy().squeeze() #(29,4)
        fmap = feaure_block[0].cpu().data.numpy().squeeze() # (29,4)

        g_cam = gard_cam(fmap, grads_val)
        array[:,i] = g_cam
    # print(array)
    node = np.mean(array,1)
    # print(array)
    # order1 = a.argsort()[::-1]  # 从大到小排序，并返回索引值
    a = node.argsort()[::-1]
    channel_num = a + 1
    print(channel_num)


