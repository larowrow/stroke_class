# train_EEGNet.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne, os, time, warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from torch.optim import lr_scheduler
from EEGNet import EEGNet

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == target).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def test(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    y_probs, y_true = [], []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            prob = torch.softmax(output, dim=1)[:, 1]
            y_probs.append(prob.cpu())
            y_true.append(target.cpu())
    auc = roc_auc_score(torch.cat(y_true), torch.cat(y_probs))
    return total_loss / len(loader), correct / len(loader.dataset), auc

# 读取数据
data_dir = "data"
sample_per_file, time_points, channels = 20, 4000, 29
X, y = [], []

for label, folder in enumerate(["mild", "severe"]):
    for file in os.listdir(os.path.join(data_dir, folder)):
        if file.endswith(".fif"):
            raw = mne.io.read_raw_fif(os.path.join(data_dir, folder, file), preload=True)
            raw.pick_types(eeg=True)
            data = raw.get_data()
            for i in range(sample_per_file):
                start, end = i * time_points, (i + 1) * time_points
                if end <= data.shape[1]:
                    segment = data[:-1, start:end]  # 去掉最后通道
                    X.append(segment)
                    y.append(label)

X, y = np.stack(X), np.array(y)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 保存测试集索引
test_indices = np.array([i for i in range(len(y)) if any(np.all(X[i] == x) for x in X_test)])
np.savetxt("test_indices.txt", test_indices, fmt='%d')

# 交叉验证训练
kf = KFold(n_splits=10, shuffle=True, random_state=42)
epochs = 150
batch_size = 32
lr, weight_decay, lr_step = 0.005, 1e-5, 70
save_path = "./eegnet_result"
os.makedirs(save_path, exist_ok=True)
# ======== 用于保存每一折数据的列表 ========
all_train_losses, all_valid_losses = [], []
all_train_accs, all_valid_accs = [], []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
    print(f"\n===== Fold {fold + 1} =====")
    model = EEGNet().to(device)
    train_loader = DataLoader(EEGDataset(X_temp[train_idx], y_temp[train_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(EEGDataset(X_temp[val_idx], y_temp[val_idx]), batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    fold_train_losses, fold_train_accs = [], []
    fold_valid_losses, fold_valid_accs = [], []
    best_auc, model_path = 0.0, os.path.join(save_path, f"best_model_fold{fold}.pth")

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_auc = test(model, val_loader, criterion)
        scheduler.step()
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_path)
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Train Acc={train_acc * 100:.2f}%, Val Acc={val_acc * 100:.2f}%, AUC={val_auc:.3f}")

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