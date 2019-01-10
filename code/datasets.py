# Create our own Pytorch DataLoader

import numpy as np
import torch
from torch.utils.data import Dataset

class DeepMotifDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLader to create batches
    """
    def __init__(self, dataset):
        """
        :param seqs: sequences where n is the number of the sample      (n,4,101)
        :param labels: all labels is 0(not bind) or 1(bind)             (n,)
        """
        super().__init__()
        seqs_np = np.asarray(dataset[0], dtype=np.float32)
        labels_np = np.asarray(dataset[1], dtype=np.float32)
        self.seqs = torch.from_numpy(seqs_np)
        self.labels = torch.from_numpy(labels_np)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq, label

    def __len__(self):
        return len(self.labels)