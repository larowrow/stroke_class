import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from EEGGCN1 import EEGGCN1
import torch
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from EEGDataset import EEGDataset
from torch_geometric.data import DenseDataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.metrics import roc_curve, roc_auc_score
import os
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
    netc = "plv"
    band = 'Gamma'
    resultpath = rootPath + "/" + t + "/" + band + "/"

    feature = np.load(resultpath + f'MI_{netc}_net_time_freq.npy')
    y = np.load(resultpath + 'y_label.npy', allow_pickle=True)
    file_path = resultpath + netc + '_network.npy'

    # 读取测试的患者ID
    f = open(resultpath + f'test_indices_{netc}.txt', 'r')
    test_indices = f.readlines()
    test_indices = [int(x.strip()) for x in test_indices]  # 将索引转为整数

    test_dataset = EEGDataset(feature=feature, y=y, indices=test_indices, root=file_path,
                              transform=Compose([ToTensor()]))
    test_dataloader = DenseDataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    all_metrics = []
    roc_save_dir = os.path.join(resultpath, "ROC_curve", "raw")
    os.makedirs(roc_save_dir, exist_ok=True)
    for fold in range(10):

        model = EEGGCN1()
        model_path = os.path.join(resultpath, "model_local", f"{netc}_kf_{fold}_local.pth")
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()

        test_loss, test_acc, y_preds, y_true, y_prods = test(testloader=test_dataloader, model=model)

        y_test_pred = [x.item() for y_pred in y_preds for x in y_pred]
        y_test_true = [x.item() for y_batch in y_true for x in y_batch]
        y_true_test = y[test_indices]

        precision_patient_test, recall_patient_test, f1_patient_test, acc_patient_test = collect_metrics(
            y_probs_test=y_prods,
            y_true_test=y_true_test,
            y_pred_test=y_test_pred,
            sample_indices_test=test_indices
        )

        # ROC & AUC
        y_probs_pos = y_prods[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true_test, y_probs_pos)
        auc = roc_auc_score(y_true_test, y_probs_pos)

        # 保存 fpr, tpr, auc
        np.save(os.path.join(roc_save_dir, f"{netc}_fpr_kf_{fold}.npy"), fpr)
        np.save(os.path.join(roc_save_dir, f"{netc}_tpr_kf_{fold}.npy"), tpr)
        np.save(os.path.join(roc_save_dir, f"{netc}_auc_kf_{fold}.npy"), np.array([auc]))


        all_metrics.append({
            'fold': fold,
            'accuracy': acc_patient_test,
            'f1': f1_patient_test,
            'recall': recall_patient_test,
            'precision': precision_patient_test,
            'auc': auc
        })

        print(
            f"[Fold {fold}] acc: {acc_patient_test:.4f}, f1: {f1_patient_test:.4f}, recall: {recall_patient_test:.4f}, precision: {precision_patient_test:.4f}, AUC: {auc:.4f}")

    # 保存全部指标到 CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(roc_save_dir, f"{netc}_metrics_all_folds.csv"), index=False)