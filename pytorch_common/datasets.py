from .datasets_dl import (DummyMultiClassDataset, DummyMultiLabelDataset, DummyRegressionDataset)


def create_dataset(dataset_name, config):
    if dataset_name == "multi_class_dataset":
        dataset = DummyMultiClassDataset(config)
    elif dataset_name == "multi_label_dataset":
        dataset = DummyMultiLabelDataset(config)
    elif dataset_name == "regression_dataset":
        dataset = DummyRegressionDataset(config)
    else:
        raise RuntimeError(f"Unknown dataset name {dataset_name}.")
    return dataset
