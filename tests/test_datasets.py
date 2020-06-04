import unittest

from pytorch_common.additional_configs import BaseDatasetConfig
from pytorch_common.datasets import create_dataset
from pytorch_common.datasets_dl import (DummyMultiClassDataset, DummyMultiLabelDataset,
                                        DummyRegressionDataset)


class TestModels(unittest.TestCase):
    def test_create_dataset(self):
        """
        Test creation of all datasets.
        """
        size, in_dim, out_dim, num_classes = 5, 4, 2, 2

        config = BaseDatasetConfig({"size": size, "dim": in_dim, "num_classes": num_classes})
        self.assertIsInstance(create_dataset("multi_class_dataset", config), DummyMultiClassDataset)

        config = BaseDatasetConfig({"size": size, "dim": in_dim, "num_classes": num_classes})
        self.assertIsInstance(create_dataset("multi_label_dataset", config), DummyMultiLabelDataset)

        config = BaseDatasetConfig({"size": size, "in_dim": in_dim, "out_dim": out_dim})
        self.assertIsInstance(create_dataset("regression_dataset", config), DummyRegressionDataset)


if __name__ == "__main__":
    unittest.main()
