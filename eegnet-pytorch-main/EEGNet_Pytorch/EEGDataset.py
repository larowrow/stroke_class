from torch.utils.data import Dataset
import numpy as np
import torch

class EEGNetDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        """
        data: shape [N, 29, 4000]
        labels: shape [N]
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # [29, 4000]
        y = self.labels[idx]
        x = np.expand_dims(x, axis=0)  # to [1, 29, 4000]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y
