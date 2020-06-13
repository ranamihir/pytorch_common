from .datasets_dl import (
    BasePyTorchDataset, DummyMultiClassDataset, DummyMultiLabelDataset, DummyRegressionDataset
)
from .additional_configs import BaseDatasetConfig


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
