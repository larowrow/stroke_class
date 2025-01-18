import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from EEGCNN import EEGCNN
import torch
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from EEGDataset import EEGDataset
import statistics as stats
from torch_geometric.data import DenseDataLoader
from torchvision.transforms import Compose, ToTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def collect_metrics(y_probs_test, y_true_test, y_pred_test, sample_indices_test):
    # 直接使用 sample_indices_test 中的索引，无需读取 CSV 文件
    rows = []
    for i in range(len(sample_indices_test)):
        idx = sample_indices_test[i]
        temp = {
            "sample_idx": idx,
            "y_true": y_true_test[i],
            "y_probs_0": y_probs_test[i, 0],
            "y_probs_1": y_probs_test[i, 1],
            "y_pred": y_pred_test[i],
        }
        rows.append(temp)

    test_patient_df = pd.DataFrame(rows)

    # 这里假设每个样本的真实标签已按顺序存储
    y_true_test_patient = y_true_test
    y_pred_test_patient = y_pred_test

    precision_patient_test = precision_score(y_true_test_patient, y_pred_test_patient)
    recall_patient_test = recall_score(y_true_test_patient, y_pred_test_patient)
    f1_patient_test = f1_score(y_true_test_patient, y_pred_test_patient)
    acc_patient_test = accuracy_score(y_true_test_patient, y_pred_test_patient)

    return precision_patient_test, recall_patient_test, f1_patient_test, acc_patient_test


def test(testloader=None, model=None):
    model.eval()
    test_loss = 0
    accuracy = 0
    y_preds = []
    y_true = []
    y_probs = torch.empty(0, 2).to(device)

    with torch.no_grad():
        for i, batch in enumerate(testloader):
            batch = batch.to(device, non_blocking=True)
            X_batch = batch
            label = torch.tensor(batch.y)

            pred = model(X_batch.x, X_batch.adj, X_batch.batch).to(device)
            loss = criterion(pred, label.to(device))
            test_loss += loss.item()
            accuracy += (torch.argmax(pred, dim=1) == label.to(device)).type(torch.float).mean().item()
            y_preds.append(torch.argmax(pred, dim=1).to(device))
            y_probs = torch.cat((y_probs, pred.data), 0)
            y_true.append(label.to(device))

    y_probs = torch.nn.functional.softmax(y_probs, dim=1).cpu().numpy()
    return test_loss / len(testloader), accuracy / len(testloader), y_preds, y_true, y_probs


if __name__ == "__main__":
    rootPath = "data"
    t = "time_8"
    netc = "pli"
    band = 'Gamma'
    resultpath = rootPath + "/" + t + "/" + band + "/"

    feature = np.load(resultpath + f'{netc}_net_time_freq.npy')
    y = np.load(resultpath + 'y_label.npy', allow_pickle=True)
    file_path = resultpath + netc + '_network.npy'

    # 读取测试的患者ID
    f = open(resultpath + f'CNN_test_indices_{netc}.txt', 'r')
    test_indices = f.readlines()
    test_indices = [int(x.strip()) for x in test_indices]  # 将索引转为整数

    test_dataset = EEGDataset(feature=feature, y=y, indices=test_indices, root=file_path,
                              transform=Compose([ToTensor()]))
    test_dataloader = DenseDataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

    precision_patient_test_folds = []
    recall_patient_test_folds = []
    f1_patient_test_folds = []
    acc_patient_test_folds = []

    model = EEGCNN()
    state_dict = torch.load(resultpath + "model_CNN" + "/" + netc + "_kf_9" + '.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()

    # 模型测试
    test_loss, test_acc, y_preds, y_true, y_prods = test(testloader=test_dataloader, model=model)
    y_test_pred = [x.item() for y_pred in y_preds for x in y_pred]
    y_test_true = [x.item() for y_test in y_true for x in y_test]

    # 真实标签根据索引直接生成
    y_true_test = y[test_indices]  # 使用 test_indices 直接获取真实标签

    precision_patient_test, recall_patient_test, f1_patient_test, acc_patient_test = collect_metrics(
        y_probs_test=y_prods, y_true_test=y_true_test, y_pred_test=y_test_pred, sample_indices_test=test_indices)

    precision_patient_test_folds.append(precision_patient_test)
    recall_patient_test_folds.append(recall_patient_test)
    f1_patient_test_folds.append(f1_patient_test)
    acc_patient_test_folds.append(acc_patient_test)

    print('acc:', acc_patient_test_folds, 'f1:', f1_patient_test_folds, 'recall:', recall_patient_test_folds,
          'precision:', precision_patient_test_folds)

    print("[MAIN] exiting...")
