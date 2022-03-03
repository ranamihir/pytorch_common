import numpy as np
from torch.utils.data import DataLoader, Dataset

from .additional_configs import BaseDatasetConfig
from .datasets_dl import BasePyTorchDataset, DummyMultiClassDataset, DummyMultiLabelDataset, DummyRegressionDataset
from .types import Optional, _Config
from .utils import setup_logging

logger = setup_logging(__name__)


def create_dataset(dataset_name: str, config: BaseDatasetConfig) -> BasePyTorchDataset:
    """
    Create and return the appropriate
    dataset from the provided config.
    """
    if dataset_name == "multi_class_dataset":
        dataset = DummyMultiClassDataset(config)
    elif dataset_name == "multi_label_dataset":
        dataset = DummyMultiLabelDataset(config)
    elif dataset_name == "regression_dataset":
        dataset = DummyRegressionDataset(config)
    else:
        raise RuntimeError(f"Unknown dataset name {dataset_name}.")
    return dataset


def create_dataloader(dataset: Dataset, config: _Config, is_train: Optional[bool] = True) -> DataLoader:
    """
    Create a dataloader wrapped
    around the given dataset.

    Option to sample a subset of the data:
    During development, you can just set
    `num_batches` or `percentage` to a small
    number to run quickly on a sample dataset.
    """
    if is_train:
        shuffle = True
        batch_size = config.train_batch_size
    else:
        shuffle = False
        batch_size = config.eval_batch_size

    num_batches, percentage = config.num_batches, config.percentage

    if num_batches and percentage:
        raise ValueError(
            f"At most one of `num_batches` ({num_batches}) or `percentage` ({percentage}) may be specified."
        )

    elif num_batches or percentage:
        n = len(dataset)
        if num_batches:
            assert num_batches <= np.ceil(n / batch_size).astype(int)
            logger.info(f"Sampling {num_batches} batches from whole dataloader.")
            sampled_indices = np.random.choice(range(n), size=min(num_batches * batch_size, n))
        else:
            assert percentage <= 100.0
            logger.info(f"Sampling {percentage}% of whole dataset.")
            sampled_indices = np.random.choice(range(n), size=int(percentage * n))

        dataset.data = dataset.data.iloc[sampled_indices].reset_index(drop=True)
        logger.info("Done.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True)

    return dataloader
