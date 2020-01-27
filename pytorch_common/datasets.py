import numpy as np
import pandas as pd
import logging

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer, BertTokenizer


class DummyMultiClassDataset(Dataset):
    def __init__(self, size=50, dim=10, n_classes=2):
        super(DummyMultiClassDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.n_classes = n_classes

    def __getitem__(self, index):
        return torch.rand(self.dim), torch.tensor(np.random.choice(range(self.n_classes)))

    def __len__(self):
        return self.size


class DummyMultiLabelDataset(Dataset):
    def __init__(self, size=50, dim=10, n_classes=5):
        super(DummyMultiLabelDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.n_classes = n_classes

    def __getitem__(self, index):
        return torch.rand(self.dim), torch.from_numpy(np.random.choice([0,1], size=self.n_classes))

    def __len__(self):
        return self.size
