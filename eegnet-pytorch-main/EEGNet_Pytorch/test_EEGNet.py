# test_EEGNet_with_fold_metrics.py
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from EEGNet import EEGNet
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import mne

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EEGDataset(Dataset):
    def __init__(self, X, y):
        # X shape expected: (N, channels, time_points)
        # EEGNet code expects input shape (batch, 1, channels, time) or similar;
        # your original code used .unsqueeze(1), 保持一致
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    result_path = "eegnet_result"
    data_dir = "../../data"
    sample_per_file, time_points, channels = 20, 4000, 29

    # ---------- load raw data (same as your original) ----------
    X, y = [], []
    for label, folder in enumerate(["mild", "severe"]):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".fif"):
                raw = mne.io.read_raw_fif(os.path.join(folder_path, file), preload=True)
                raw.pick_types(eeg=True)
                data = raw.get_data()
                for i in range(sample_per_file):
                    start, end = i * time_points, (i + 1) * time_points
                    if end <= data.shape[1]:
                        segment = data[:-1, start:end]  # 去掉最后通道（你原本这样做）
                        X.append(segment)
                        y.append(label)

    X = np.stack(X)   # shape: (N, channels-1, time_points)
    y = np.array(y)

    # ---------- 载入 test indices ----------
    test_indices = np.loadtxt("test_indices.txt", dtype=int)
    X_test, y_test = X[test_indices], y[test_indices]

    test_loader = DataLoader(EEGDataset(X_test, y_test), batch_size=32, shuffle=False)

    all_preds = []
    all_probs = []
    fold_metrics = []

    for fold in range(10):
        model = EEGNet().to(device)
        model_file = os.path.join(result_path, f"best_model_fold{fold}.pth")

        # 更稳健地加载模型：考虑到 CPU/GPU 情况
        state = torch.load(model_file, map_location=device)
        model.load_state_dict(state)
        model.eval()

        preds, probs = [], []

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output = model(data)  # 形状 (batch, n_classes)
                prob = torch.softmax(output, dim=1)[:, 1].cpu().numpy()  # 类别1概率
                pred = output.argmax(dim=1).cpu().numpy()
                probs.extend(prob.tolist())
                preds.extend(pred.tolist())

        # 检查预测长度是否和 y_test 一致
        assert len(preds) == len(y_test), f"Fold {fold}: preds length {len(preds)} != y_test length {len(y_test)}"

        # 计算每折指标，处理可能的异常情况
        acc_fold = accuracy_score(y_test, preds)
        f1_fold = f1_score(y_test, preds, zero_division=0)
        recall_fold = recall_score(y_test, preds, zero_division=0)
        precision_fold = precision_score(y_test, preds, zero_division=0)
        try:
            auc_fold = roc_auc_score(y_test, probs)
        except Exception:
            auc_fold = np.nan

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": acc_fold,
            "F1": f1_fold,
            "Recall": recall_fold,
            "Precision": precision_fold,
            "AUC": auc_fold
        })

        all_preds.append(preds)
        all_probs.append(probs)

        print(f"Fold {fold} done: Acc={acc_fold:.4f} F1={f1_fold:.4f} AUC={np.nan if np.isnan(auc_fold) else auc_fold:.4f}")

    # ---------- 保存每折指标到 CSV，并追加 mean/std 行 ----------
    df_metrics = pd.DataFrame(fold_metrics)
    metrics_cols = ["Accuracy", "F1", "Recall", "Precision", "AUC"]
    means = df_metrics[metrics_cols].mean()
    stds = df_metrics[metrics_cols].std()

    summary_rows = pd.DataFrame([["mean"] + means.tolist(), ["std"] + stds.tolist()],
                                columns=["Fold"] + metrics_cols)
    df_out = pd.concat([df_metrics, summary_rows], ignore_index=True)
    out_csv = os.path.join(result_path, "fold_metrics.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"每折指标已保存到: {out_csv}")

    # ---------- 多数投票 & 概率平均（原有逻辑） ----------
    all_preds = np.array(all_preds).T
    all_probs = np.array(all_probs).T
    final_preds = [np.bincount(p).argmax() for p in all_preds]
    final_probs = np.mean(all_probs, axis=1)

    # 最终指标
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds, zero_division=0)
    recall = recall_score(y_test, final_preds, zero_division=0)
    precision = precision_score(y_test, final_preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, final_probs)
    except Exception:
        auc = np.nan
    fpr, tpr, _ = roc_curve(y_test, final_probs)

    print("==== 测试集指标（多数投票） ====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC:       {auc:.4f}")

    # 保存最终 summary
    with open(os.path.join(result_path, "test_metrics_summary.txt"), "w") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"AUC:       {auc:.4f}\n")

    np.save(os.path.join(result_path, "final_fpr.npy"), fpr)
    np.save(os.path.join(result_path, "final_tpr.npy"), tpr)

    # ROC 可视化（如需保存改注释）
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
