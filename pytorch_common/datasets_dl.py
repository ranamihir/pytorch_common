from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from types import FunctionType

import torch
from torch.utils.data import Dataset

from .utils import save_object, load_object, remove_object, print_dataframe
from .types import Callable, Optional, _StringDict


class BasePyTorchDataset(Dataset):
    """
    Generic PyTorch Dataset implementing useful methods.
    This dataset makes a few assumptions:
      - The final data is stored in `self.data`
      - `self.data` is a pandas dataframe
      - The target column is `"target"`.
        But you may set it to the desired column during
        initialization, and rest should work as-is.
    """
    def __init__(self):
        """
        Initialize BasePyTorchDataset.
          - Set the target column
          - Enable tqdm
        """
        super().__init__()
        self.__name__ = self.__class__.__name__ # Set dataset name
        self.target_col = "target"
        tqdm.pandas() # Enable tqdm

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def print_dataset(self) -> None:
        """
        Print useful summary statistics of the dataset.
        """
        logging.info("\n"+"-"*40)
        print_dataframe(self.data)
        value_counts = self.data[self.target_col].value_counts()
        logging.info(f"Target value counts:\n{value_counts}")
        logging.info("\n"+"-"*40)

    def save(self, *args, **kwargs) -> None:
        """
        Save the dataset.
        This method only saves the object attributes. If the
        underlying methods are modified, you might need to
        re-serialize the object and then save it again.

        Note: See `utils.save_object()` for available arguments.
        """
        save_object(self, *args, **kwargs)

    @classmethod
    def load(cls, *args, **kwargs) -> BasePyTorchDataset:
        """
        Load the dataset.
        This method only loads the object attributes. If the
        underlying methods are modified, you might need to save
        the object after re-serialization and then load it again.

        Note: See `utils.load_object()` for available arguments.
        """
        dataset = load_object(*args, **kwargs)
        return dataset

    @classmethod
    def remove(cls, *args, **kwargs) -> None:
        """
        Remove the dataset at the given path.

        Note: See `utils.remove_object()` for available arguments.
        """
        remove_object(*args, **kwargs)

    def progress_apply(self, data: pd.DataFrame, func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Generic function to `progress_apply` a given row-level
        function `func` on the given `data` (chunk).
        """
        return data.progress_apply(func, *args, **kwargs, axis=1)

    def oversample_class(self, class_to_oversample: Optional[int] = None) -> pd.DataFrame:
        """
        Oversample the given class.
        `self.oversampling_factor` must
        be set to use this method.
        The final count of the class will be
        (original count) * `oversampling_factor`.
        :param class_to_oversample: Class (label) to oversample.
                                    Oversamples minority class
                                    by default.
        """
        # Get appropriate class label and count
        class_label, class_count = self._get_class_count(class_to_oversample, minority=True)

        # Randomly sample indices to oversample
        class_indices = self.data[self.data[self.target_col] == class_label].index.tolist()
        num_to_oversample = np.floor(class_count * (self.oversampling_factor-1)).astype(int)
        indices = np.random.choice(class_indices, size=num_to_oversample, replace=True)

        # Append oversampled rows at the bottom and shuffle data
        self.data = self.data.append(self.data.iloc[indices])
        self.shuffle_and_reindex_data()

    def undersample_class(self, class_to_undersample: Optional[int] = None) -> pd.DataFrame:
        """
        Undersample the given class.
        `self.undersampling_factor` must
        be set to use this method.
        The final count of the class will be
        (original count) / `undersampling_factor`.
        :param class_to_undersample: Class (label) to undersample.
                                     Undersamples majority class
                                     by default.
        """
        # Get appropriate class label and count
        class_label, class_count = self._get_class_count(class_to_undersample, minority=False)

        # Randomly sample indices to undersample
        class_indices = self.data[self.data[self.target_col] == class_label].index.tolist()
        num_to_remove = np.floor(class_count*(1 - 1/self.undersampling_factor)).astype(int)
        indices = np.random.choice(class_indices, size=num_to_remove, replace=False)

        # Remove undersampled rows and shuffle data
        self.data.drop(index=indices, inplace=True)
        self.shuffle_and_reindex_data()

    def _get_class_count(
        self,
        class_to_sample: Optional[int] = None,
        minority: bool = True
    ) -> Tuple[int]:
        """
        Get the counts of each class.
        Used for under-/over-sampling.
        """
        # Get all class counts
        value_counts = self.data[self.target_col].value_counts(sort=True, ascending=False)

        # If class not specified, take majority/minority class by default
        class_label = class_to_sample
        if class_label is None:
            class_label = value_counts.index.tolist()[-1 if minority else 0]
        class_count = value_counts[class_label]
        return class_label, class_count

    def shuffle_and_reindex_data(self) -> None:
        """
        Shuffle and reindex data.
        """
        # Shuffle and reindex data
        self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __getstate__(self) -> _StringDict:
        """
        Update `__getstate__` to exclude object
        methods so that it can be pickled.
        """
        return {k: v for k, v in self.__dict__.items() if not isinstance(v, FunctionType)}


class DummyMultiClassDataset(BasePyTorchDataset):
    """
    Dummy dataset for generating
    random multi-class style data.
    """
    def __init__(self, config: BasePyTorchDataset):
        super().__init__()
        self.size = config.size
        self.dim = config.dim
        self.num_classes = config.num_classes

        # Create random data
        self.data = create_dataframe(self.size, self.target_col, self._get_row)

    def __getitem__(self, index):
        return self.data.iloc[index].tolist()

    def __len__(self):
        return self.size

    def _get_row(self, index: int) -> pd.Series:
        """
        Generate data for each row.
        Fix torch random generator seed and
        numpy random state for reproducibility.
        """
        x = torch.rand(self.dim, generator=torch.Generator().manual_seed(index))
        y = torch.as_tensor(np.random.RandomState(index).choice(range(self.num_classes)))
        return pd.Series([x.float(), y.long()])


class DummyMultiLabelDataset(BasePyTorchDataset):
    """
    Dummy dataset for generating
    random multi-label style data.
    """
    def __init__(self, config: BasePyTorchDataset):
        super().__init__()
        self.size = config.size
        self.dim = config.dim
        self.num_classes = config.num_classes

        # Create random data
        self.data = create_dataframe(self.size, self.target_col, self._get_row)

    def __getitem__(self, index):
        return self.data.iloc[index].tolist()

    def __len__(self):
        return self.size

    def _get_row(self, index: int) -> pd.Series:
        """
        Generate data for each row.
        Fix torch random generator seed and
        numpy random state for reproducibility.
        """
        x = torch.rand(self.dim, generator=torch.Generator().manual_seed(index))
        y = torch.as_tensor(np.random.RandomState(index).choice([0,1], size=self.num_classes))
        return pd.Series([x.float(), y.long()])


class DummyRegressionDataset(BasePyTorchDataset):
    """
    Dummy dataset for generating
    random regression style data.
    """
    def __init__(self, config: BasePyTorchDataset):
        super().__init__()
        self.size = config.size
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim

        # Create random data
        self.data = create_dataframe(self.size, self.target_col, self._get_row)

    def __getitem__(self, index):
        return self.data.iloc[index].tolist()

    def __len__(self):
        return self.size

    def _get_row(self, index: int) -> pd.Series:
        """
        Generate data for each row.
        Fix torch random generator seed and
        numpy random state for reproducibility.
        """
        x = torch.rand(self.in_dim, generator=torch.Generator().manual_seed(index))
        y = torch.as_tensor(np.random.RandomState(index).randn(self.out_dim))
        return pd.Series([x.float(), y.float()])


def create_dataframe(
    size: int,
    target_col: str,
    row_gen_func: Callable[[int], pd.Series]
) -> pd.DataFrame:
    """
    Generate a dataframe of length `size` using
    the function `row_gen_func` to generate the
    data for each row.
    """
    columns = ["features", target_col]
    data = pd.DataFrame(columns=columns, index=range(size))
    data[columns] = data.index.to_series().apply(row_gen_func)
    return data
