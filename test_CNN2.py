import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from EEGCNN2 import EEGCNN2
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, roc_curve
import os
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


if __name__ == "__main__":
    rootPath = "data"
    time_w = "time_8"
    band = 'Gamma'
    connetc = "plv"
    resultpath = os.path.join(rootPath, time_w, band, "CNN2_result")

    feature = np.load(os.path.join(rootPath, time_w, band, f'MI_{connetc}_net_time_freq.npy'))
    y = np.load(os.path.join(rootPath, time_w, band, 'y_label.npy'))
    test_indices = np.loadtxt(os.path.join(resultpath, "test_indices.txt"), dtype=int)

    X_test = feature[test_indices]
    y_test = y[test_indices]
    test_loader = DataLoader(FeatureDataset(X_test, y_test), batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []
    fold_metrics = []  # 用于保存每一折的指标

    for fold in range(10):
        model = EEGCNN2().to(device)
        model_path = os.path.join(resultpath, f"best_model_fold{fold}.pth")
        model.load_state_dict(torch.load(model_path))
        model.eval()

        fold_preds = []
        fold_probs = []

        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                output = model(x.view(x.shape[0], 29, 12), None, None)
                prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # 类别1概率
                pred = output.argmax(dim=1).cpu().numpy()
                fold_preds.extend(pred)
                fold_probs.extend(prob)

        # 每折指标
        acc = accuracy_score(y_test, fold_preds)
        f1 = f1_score(y_test, fold_preds)
        recall = recall_score(y_test, fold_preds)
        precision = precision_score(y_test, fold_preds)
        auc = roc_auc_score(y_test, fold_probs)

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": acc,
            "F1": f1,
            "Recall": recall,
            "Precision": precision,
            "AUC": auc
        })

        all_preds.append(fold_preds)
        all_probs.append(fold_probs)

    # 保存每折指标到 CSV
    df_metrics = pd.DataFrame(fold_metrics)
    df_metrics.to_csv(os.path.join(resultpath, "fold_metrics.csv"), index=False)
    print(f"每折指标已保存到 {os.path.join(resultpath, 'fold_metrics.csv')}")

    # 多数投票 + 平均概率
    all_preds = np.array(all_preds).T
    all_probs = np.array(all_probs).T
    final_preds = [np.bincount(row).argmax() for row in all_preds]
    final_probs = np.mean(all_probs, axis=1)

    # 最终综合指标
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds)
    auc = roc_auc_score(y_test, final_probs)
    fpr, tpr, _ = roc_curve(y_test, final_probs)

    print("\n==== 最终综合结果（多数投票） ====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC:       {auc:.4f}")

    # 保存综合指标
    with open(os.path.join(resultpath, "test_metrics_summary.txt"), "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"AUC:       {auc:.4f}\n")

    # 保存 ROC
    np.save(os.path.join(resultpath, "fpr.npy"), fpr)
    np.save(os.path.join(resultpath, "tpr.npy"), tpr)
    np.save(os.path.join(resultpath, "test_probs.npy"), final_probs)

    # # 绘制 ROC
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend(loc="lower right")
    # plt.grid(True)
    # plt.show()
