import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, BatchNorm
from torch_geometric.data import DenseDataLoader
from torchvision.transforms import Compose, ToTensor
from sklearn.model_selection import train_test_split, KFold
import shap
import matplotlib.pyplot as plt
import pandas as pd
from EEGDataset import EEGDataset
# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# EEGGCN1 模型
# -------------------------------
class EEGGCN1(nn.Module):
    def __init__(self):
        super(EEGGCN1, self).__init__()
        torch.manual_seed(42)
        self.width = 14
        self.conv1 = DenseGCNConv(13, self.width)
        self.batchnorm1 = BatchNorm(self.width, eps=1e-5, momentum=0.1)
        self.linear1 = nn.Linear(29 * self.width, 29)
        self.linear2 = nn.Linear(29, 2)

    def forward(self, f, adj, batch):
        out = self.conv1(f, adj)
        out = out.view(-1, self.width)
        out = F.dropout(F.leaky_relu(self.batchnorm1(out), 0.01), p=0.7, training=self.training)
        out = out.view(-1, 29, self.width)
        out = out.view(-1, 29 * self.width)
        out = F.dropout(F.leaky_relu(self.linear1(out), 0.01), p=0.7, training=self.training)
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


# -------------------------------
# SHAP Explainer
# -------------------------------
class EEGGCN1_SHAP_Explainer:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.model.eval()
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(13)]

    def predict(self, X_features, X_adj):
        """
        X_features: (batch_size, num_nodes, num_features)
        X_adj: (batch_size, num_nodes, num_nodes)
        """
        if isinstance(X_features, np.ndarray):
            X_features = torch.FloatTensor(X_features).to(device)
        if isinstance(X_adj, np.ndarray):
            X_adj = torch.FloatTensor(X_adj).to(device)

        with torch.no_grad():
            outputs = self.model(X_features, X_adj, torch.arange(X_features.shape[0]).repeat_interleave(X_features.shape[1]).to(device))
        return outputs.cpu().numpy()

    def explain(self, X_background, X_background_adj, X_explain, X_explain_adj, class_index=1):
        explainer = shap.DeepExplainer(
            model=lambda x: torch.tensor(self.predict(x, X_explain_adj), dtype=torch.float32),
            data=torch.FloatTensor(X_background).to(device)
        )
        shap_values = explainer.shap_values(torch.FloatTensor(X_explain).to(device))
        return shap_values[class_index] if isinstance(shap_values, list) else shap_values

    def visualize_shap(self, shap_values, X_explain, sample_idx=0, save_path=None):
        sample_shap = shap_values[sample_idx]
        sample_data = X_explain[sample_idx]
        mean_abs_shap = np.mean(np.abs(sample_shap), axis=0)

        plt.figure(figsize=(12, 8))
        # 特征重要性
        plt.subplot(2, 1, 1)
        indices = np.argsort(mean_abs_shap)[::-1]
        sorted_features = [self.feature_names[i] for i in indices]
        plt.bar(range(len(indices)), mean_abs_shap[indices])
        plt.xticks(range(len(indices)), sorted_features, rotation=45)
        plt.title("Feature Importance (Mean |SHAP|)")
        plt.tight_layout()

        # 节点级热图
        plt.subplot(2, 1, 2)
        im = plt.imshow(sample_shap, aspect='auto', cmap='RdBu_r')
        plt.colorbar(im, label='SHAP Value')
        plt.xlabel('Features')
        plt.ylabel('Nodes')
        plt.title('Node-level SHAP Values')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return mean_abs_shap


# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    # 参数
    rootPath = "data"
    time_w = "time_8"
    band = "Gamma_LI"
    connetc = "plv"
    resultpath = os.path.join(rootPath, time_w, band)
    feature = np.load(os.path.join(resultpath, f'MI_{connetc}_net_time_freq.npy'))
    adj_all = np.load(os.path.join(resultpath, f'{connetc}_network.npy'))  # shape (N, 29, 29)
    y = np.load(os.path.join(resultpath, 'y_label.npy'), allow_pickle=True)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 训练/测试划分
    num_samples_per_class = 360
    total_samples = 2 * num_samples_per_class
    train_indices, test_indices = train_test_split(
        np.arange(total_samples),
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )
    with open(os.path.join(resultpath, f'test_indices_{connetc}.txt'), 'w') as f:
        for idx in test_indices:
            f.write(f"{idx}\n")

    # KFold
    skf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    for fold, (train_id_idx, val_id_idx) in enumerate(skf.split(train_indices)):
        print(f"==== Fold {fold+1} ====")
        train_dataset = EEGDataset(feature=feature, y=y, indices=train_id_idx, root=os.path.join(resultpath, f'{connetc}_network.npy'), transform=Compose([ToTensor()]))
        val_dataset = EEGDataset(feature=feature, y=y, indices=val_id_idx, root=os.path.join(resultpath, f'{connetc}_network.npy'), transform=Compose([ToTensor()]))

        train_loader = DenseDataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DenseDataLoader(val_dataset, batch_size=32, shuffle=False)

        # 模型
        model = EEGGCN1().to(device)
        model.eval()

        # -----------------------
        # SHAP解释示例
        # -----------------------
        # 假设要从测试集采样
        test_dataset = EEGDataset(feature=feature, y=y, indices=test_indices, root=os.path.join(resultpath, f'{connetc}_network.npy'), transform=Compose([ToTensor()]))
        test_loader = DenseDataLoader(test_dataset, batch_size=32, shuffle=False)

        X_explain = []
        X_explain_adj = []
        y_explain = []

        for batch in test_loader:
            batch = batch.to(device)
            f_batch = batch.x.cpu().numpy().reshape(-1, 29, 12)
            adj_batch = batch.adj.cpu().numpy().reshape(-1, 29, 29)  # EEGDataset需返回adj
            labels_batch = batch.y.cpu().numpy()
            X_explain.extend(f_batch)
            X_explain_adj.extend(adj_batch)
            y_explain.extend(labels_batch)

        X_explain = np.array(X_explain)
        X_explain_adj = np.array(X_explain_adj)
        y_explain = np.array(y_explain)

        # 背景数据（轻症）
        mild_idx = np.where(y_explain == 0)[0][:100]
        X_background = X_explain[mild_idx]
        X_background_adj = X_explain_adj[mild_idx]

        feature_names = ['mav','std','betweenness','psd','PersistenceEntropy_0','PersistenceEntropy_1',
                         'landscape1_0','landscape1_1','landscape2_0','landscape2_1','betti_0','betti_1','LI']

        explainer = EEGGCN1_SHAP_Explainer(model, feature_names=feature_names)
        shap_values = explainer.explain(X_background, X_background_adj, X_explain, X_explain_adj, class_index=1)
        explainer.visualize_shap(shap_values, X_explain, sample_idx=0)
