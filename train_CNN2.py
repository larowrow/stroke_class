# train_EEGCNN2.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
import os
from EEGCNN2 import EEGCNN2
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x.view(x.shape[0], 29, 12), None, None)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(dim=1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def test(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x.view(x.shape[0], 29, 12), None, None)
            loss = criterion(output, y)
            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

if __name__ == "__main__":
    rootPath = "data"
    time_w = "time_8"
    band = 'Gamma'
    connetc = "plv"
    resultpath = os.path.join(rootPath, time_w, band)
    save_path = os.path.join(resultpath, "CNN2_result")
    os.makedirs(save_path, exist_ok=True)

    feature = np.load(os.path.join(resultpath, f'MI_{connetc}_net_time_freq.npy'))
    y = np.load(os.path.join(resultpath, 'y_label.npy'))

    indices = np.arange(len(feature))
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        feature, y, indices, test_size=0.2, random_state=42)


    np.savetxt(os.path.join(save_path, "test_indices.txt"), idx_test, fmt="%d")
    # np.save(os.path.join(save_path, "X_test.npy"), X_test)
    # np.save(os.path.join(save_path, "y_test.npy"), y_test)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    batch_size = 32
    lr = 0.005
    epochs = 150
    weight_decay = 1e-4
    lr_step = 50

    # ======== 用于保存每一折数据的列表 ========
    all_train_losses, all_valid_losses = [], []
    all_train_accs, all_valid_accs = [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
        print(f"\n======== Fold {fold+1} ========")
        model = EEGCNN2().to(device)

        train_loader = DataLoader(FeatureDataset(X_temp[train_idx], y_temp[train_idx]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FeatureDataset(X_temp[val_idx], y_temp[val_idx]), batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        fold_train_losses, fold_train_accs = [], []
        fold_valid_losses, fold_valid_accs= [], []
        best_acc = 0.0
        best_model_path = os.path.join(save_path, f"best_model_fold{fold}.pth")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            val_loss, val_acc = test(model, val_loader, criterion)
            scheduler.step()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            fold_train_losses.append(train_loss)
            fold_train_accs.append(train_acc)
            fold_valid_losses.append(val_loss)
            fold_valid_accs.append(val_acc)

        # 汇总每一折的数据
        all_train_losses.append(fold_train_losses)
        all_train_accs.append(fold_train_accs)
        all_valid_losses.append(fold_valid_losses)
        all_valid_accs.append(fold_valid_accs)

    # ======== 训练结束后统一保存 ========
    np.save(os.path.join(save_path, "train_losses.npy"), all_train_losses)
    np.save(os.path.join(save_path, "train_accs.npy"), all_train_accs)
    np.save(os.path.join(save_path, "valid_losses.npy"), all_valid_losses)
    np.save(os.path.join(save_path, "valid_accs.npy"), all_valid_accs)

