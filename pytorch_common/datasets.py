import numpy as np
import pandas as pd
import logging

import torch
from torch.utils.data import Dataset


class DummyMultiClassDataset(Dataset):
    def __init__(self, size=50, dim=10, num_classes=2):
        super().__init__()
        self.size = size
        self.dim = dim
        self.num_classes = num_classes

    def __getitem__(self, index):
        # Fix torch random generator seed and numpy random state for reproducibility
        x = torch.rand(self.dim, generator=torch.Generator().manual_seed(index))
        y = torch.as_tensor(np.random.RandomState(index).choice(range(self.num_classes)))
        return x, y

    def __len__(self):
        return self.size


class DummyMultiLabelDataset(Dataset):
    def __init__(self, size=50, dim=10, num_classes=5):
        super().__init__()
        self.size = size
        self.dim = dim
        self.num_classes = num_classes

    def __getitem__(self, index):
        # Fix torch random generator seed and numpy random state for reproducibility
        x = torch.rand(self.dim, generator=torch.Generator().manual_seed(index))
        y = torch.as_tensor(np.random.RandomState(index).choice([0,1], size=self.num_classes))

    def __len__(self):
        return self.size
