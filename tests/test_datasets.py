import unittest

import numpy as np

from pytorch_common.additional_configs import BaseDatasetConfig
from pytorch_common.datasets import create_dataset
from pytorch_common.datasets_dl import (
    BasePyTorchDataset,
    DummyMultiClassDataset,
    DummyMultiLabelDataset,
    DummyRegressionDataset,
)
from pytorch_common.types import Optional, _StringDict


class TestModels(unittest.TestCase):
    @classmethod
    def setUp(cls):
        """
        Set up default config dictionary for a dataset.
        """
        cls.default_config_dict = {"size": 30, "dim": 4, "num_classes": 3}

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

    def test_oversample_class(self):
        """
        Test oversampling the targets of a dataset.
        """
        oversampling_factor = 2
        dataset = self._get_dataset(dictionary={"oversampling_factor": oversampling_factor})

        # Get original class info
        minority_info_pre = dataset._get_class_info(minority=True)
        majority_class_pre = dataset._get_class_info(minority=False)

        # Oversample minority class
        dataset.sample_class(oversampling_factor=oversampling_factor)

        # Get class info after oversampling
        # Ensure consistency in labels (order might have changed after sampling)
        minority_class_post = dataset._get_class_info(class_to_sample=minority_info_pre["label"])
        majority_class_post = dataset._get_class_info(class_to_sample=majority_class_pre["label"])

        # Ensure correct number of rows for both classes
        self.assertEqual(minority_class_post["count"], minority_info_pre["count"] * oversampling_factor)
        self.assertEqual(majority_class_post["count"], majority_class_pre["count"])

    def test_undersample_class(self):
        """
        Test undersampling the targets of a dataset.
        """
        undersampling_factor = 2
        dataset = self._get_dataset(dictionary={"undersampling_factor": undersampling_factor})

        # Get original class info
        minority_info_pre = dataset._get_class_info(minority=True)
        majority_class_pre = dataset._get_class_info(minority=False)

        # Undersample minority class
        dataset.sample_class(undersampling_factor=undersampling_factor)

        # Get class info after undersampling
        # Ensure consistency in labels (order might have changed after sampling)
        minority_class_post = dataset._get_class_info(class_to_sample=minority_info_pre["label"])
        majority_class_post = dataset._get_class_info(class_to_sample=majority_class_pre["label"])

        # Ensure correct number of rows for both classes
        self.assertEqual(minority_class_post["count"], minority_info_pre["count"])
        self.assertEqual(
            majority_class_post["count"], np.ceil(majority_class_pre["count"] / undersampling_factor),
        )

    def _get_dataset(
        self, dataset_name: Optional[str] = "multi_class_dataset", dictionary: Optional[_StringDict] = None,
    ) -> BasePyTorchDataset:
        """
        Merge the provided `dictionary` with the default
        one and create and return the desired dataset.
        """
        if dictionary is None:
            dictionary = {}
        dictionary = {**self.default_config_dict, **dictionary}
        return create_dataset(dataset_name, BaseDatasetConfig(dictionary))


if __name__ == "__main__":
    unittest.main()
