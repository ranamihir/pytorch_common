import unittest
import torch

from pytorch_common.config import Config, load_pytorch_common_config, set_pytorch_config
from pytorch_common import utils
from pytorch_common.types import Dict, Optional


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up config with default parameters.
        """
        cls.default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cls.config = {
            "load_pytorch_common_config": True,
            "transientdir": "dummy_transientdir",
            "packagedir": "dummy_package_dir",
            "misc_data_dir": "dummy_misc_data_dir",
            "device": cls.default_device
        }

        cls.n_gpu = torch.cuda.device_count()

    @classmethod
    def tearDownClass(cls):
        """
        Delete all directories created during config initialization.
        """
        for directory in ["transientdir", "packagedir", "misc_data_dir"]:
            utils.remove_dir(cls.config[directory], force=True)

    def test_load_pytorch_common_config(self):
        """
        Test loading of config with different configurations.
        """
        config = self._load_config({"classification_type": "multiclass"})
        self.assertEqual(config.classification_type, "multiclass") # Test overridden value
        self.assertEqual(config.device, self.default_device) # Test overridden value
        self.assertFalse(config.assert_gpu) # Test same value

        # Test disabling loading of pytorch_common config
        dictionary = {"load_pytorch_common_config": False}
        self.assertEqual(dict(self._load_config(dictionary)), self._get_merged_dict(dictionary))

    def test_set_loss_and_eval_criteria(self):
        """
        Test different loss/eval criteria configurations.
        """
        # Test incompatible settings
        self._test_config_error({"model_type": "regression", "loss_criterion": "accuracy"})
        self._test_config_error({"model_type": "classification", "eval_criteria": ["accuracy"],
                                 "use_early_stopping": False, "early_stopping_criterion": "f1"})
        self._test_config_error({"model_type": "classification", "eval_criteria": ["accuracy"],
                                 "use_early_stopping": True, "early_stopping_criterion": "f1"})
        self._test_config_error({"model_type": "classification", "classification_type": "dummy"})
        for classification_type in ["binary", "multiclass", "multilabel"]:
            self._test_config_error({"model_type": "regression",
                                     "classification_type": classification_type})

        # Test that FocalLoss only compatible with binary classification
        self._load_config({"model_type": "classification", "classification_type": "binary",
                           "loss_criterion": "focal-loss"})
        for classification_type in ["multiclass", "multilabel"]:
            self._test_config_error({"model_type": "classification",
                                     "classification_type": classification_type,
                                     "loss_criterion": "focal-loss"})

    def test_check_and_set_devices(self):
        """
        Test GPU/CPU device configuration.
        """
        # Test incompatible settings
        self._test_config_error({"device": "cpu", "device_ids": [0,1]})
        self.assertEqual(self._load_config({"device": "cpu"}).n_gpu, 0)

        if torch.cuda.is_available():
            # Test that `device_ids` are automatically swapped if specified order is incorrect
            config = self._load_config({"device": "cuda:0", "device_ids": [1,0]})
            self.assertEqual(config.device_ids, [0,1])

            if self.n_gpu > 1:
                # Test full parallelization if `device_ids==-1`
                config = self._load_config({"device": "cuda:0", "device_ids": -1})
                self.assertEqual(config.device_ids, list(range(self.n_gpu)))

    def test_set_batch_size(self):
        """
        Test batch size configuration.
        """
        # Ensure final derived batch size is correct
        self.assertEqual(self._load_config({"batch_size": 8}).batch_size, 8)
        self.assertEqual(self._load_config({"device": "cpu", "batch_size": 8}).batch_size, 8)

        # Test incompatible setting
        self._test_config_error({"batch_size": 8, "batch_size_per_gpu": 8}, error=ValueError)

        if torch.cuda.is_available():
            # Ensure final derived batch size is correct
            config = self._load_config({"device": "cuda:0", "device_ids": [0],
                                        "batch_size_per_gpu": 8})
            self.assertEqual(config.batch_size, 8)

            if self.n_gpu > 1:
                # Ensure final derived batch size is correct
                config = self._load_config({"device": "cuda:0", "device_ids": -1,
                                            "batch_size_per_gpu": 8})
                self.assertEqual(config.batch_size, 8*self.n_gpu)

    def _test_config_error(self, dictionary: Dict, error=AssertionError):
        """
        Generic code to assert that `error` is raised when
        loading config with an overriding `dictionary`.
        """
        with self.assertRaises(error):
            self._load_config(dictionary)

    def _load_config(self, dictionary: Optional[Dict] = None) -> Config:
        """
        Load the default pytorch_common config
        after overriding it with `dictionary`.
        """
        dictionary = self._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        return config

    def _get_merged_dict(self, dictionary: Optional[Dict] = None) -> Dict:
        """
        Override default config with
        `dictionary` if provided.
        """
        if dictionary is None:
            return self.config.copy()
        return {**self.config, **dictionary}


if __name__ == "__main__":
    unittest.main()
