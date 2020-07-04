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
            "batch_size_per_gpu": 8,
            "device": cls.default_device,
        }

        cls.n_gpu = torch.cuda.device_count()

        cls.base_batch_size = cls.config["batch_size_per_gpu"]

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
        self.assertEqual(config.classification_type, "multiclass")  # Test overridden value
        self.assertEqual(config.device, self.default_device)  # Test overridden value
        self.assertFalse(config.assert_gpu)  # Test same value

        # Test disabling loading of pytorch_common config
        dictionary = {"load_pytorch_common_config": False}
        self.assertEqual(dict(self._load_config(dictionary)), self._get_merged_dict(dictionary))

    def test_set_loss_and_eval_criteria(self):
        """
        Test different loss/eval criteria configurations.
        """
        # Test incompatible settings
        self._test_config_error({"model_type": "regression", "loss_criterion": "accuracy"})
        self._test_config_error(
            {
                "model_type": "classification",
                "eval_criteria": ["accuracy"],
                "use_early_stopping": False,
                "early_stopping_criterion": "f1",
            }
        )
        self._test_config_error(
            {
                "model_type": "classification",
                "eval_criteria": ["accuracy"],
                "use_early_stopping": True,
                "early_stopping_criterion": "f1",
            }
        )
        self._test_config_error({"model_type": "classification", "classification_type": "dummy"})
        for classification_type in ["binary", "multiclass", "multilabel"]:
            self._test_config_error({"model_type": "regression", "classification_type": classification_type})

        # Test that FocalLoss only compatible with binary classification
        self._load_config(
            {"model_type": "classification", "classification_type": "binary", "loss_criterion": "focal-loss",}
        )
        for classification_type in ["multiclass", "multilabel"]:
            self._test_config_error(
                {
                    "model_type": "classification",
                    "classification_type": classification_type,
                    "loss_criterion": "focal-loss",
                }
            )

    def test_check_and_set_devices_on_cpu(self):
        """
        Test CPU device configuration.
        """
        # Test incompatible settings
        self._test_config_error({"device": "cpu", "device_ids": [0, 1]})
        self.assertEqual(self._load_config({"device": "cpu"}).n_gpu, 0)

    def test_check_and_set_devices_on_gpu(self):
        """
        Test GPU device configuration.
        """
        if torch.cuda.is_available():
            # Test that `device_ids` are automatically swapped if specified order is incorrect
            config = self._load_config({"device": "cuda:0", "device_ids": [1, 0]})
            self.assertEqual(config.device_ids, [0, 1])

            if self.n_gpu > 1:
                # Test full parallelization if `device_ids==-1`
                config = self._load_config({"device": "cuda:0", "device_ids": -1})
                self.assertEqual(config.device_ids, list(range(self.n_gpu)))
        else:
            raise unittest.SkipTest("GPU(s) not available.")

    def test_deprecated_batch_size(self):
        """
        Test that providing deprecated
        arugments raises an error.
        """
        # Test deprecated arugments
        self._test_config_error({"device": "cpu", "batch_size_per_gpu": None, "batch_size": 8}, error=ValueError)
        self._test_config_error({"batch_size_per_gpu": None, "batch_size": 8}, error=ValueError)

    def test_incompatible_batch_sizes(self):
        """
        Test that mutually incompatible
        configuration of batch size raises
        an error.
        """
        # No batch size provided
        self._test_config_error({"batch_size_per_gpu": None}, error=ValueError)

        SUPPORTED_MODES = ["train", "eval", "test"]
        batch_size_dict = {f"{mode}_batch_size_per_gpu": self.base_batch_size for mode in SUPPORTED_MODES}

        # Raise error if both a common and mode-specific `batch_size_per_gpu` is provided
        self._test_config_error({**batch_size_dict}, error=ValueError)

    def test_set_all_batch_sizes_on_cpu(self):
        """
        Test batch size configuration on CPU.
        """
        SUPPORTED_MODES = ["train", "eval", "test"]
        batch_size_dict = {
            f"{mode}_batch_size_per_gpu": self.base_batch_size * (i + 1) for i, mode in enumerate(SUPPORTED_MODES)
        }

        # Ensure final derived batch size for each mode on CPU is correct
        same_batch_size_dict = {"device": "cpu"}
        different_batch_sizes_dict = {
            "device": "cpu",
            "batch_size_per_gpu": None,
            **batch_size_dict,
        }
        self._test_all_batch_sizes(same_batch_size_dict, different_batch_sizes_dict, 1)

    def test_set_all_batch_sizes_on_gpu(self):
        """
        Test batch size configuration on GPU.
        """
        SUPPORTED_MODES = ["train", "eval", "test"]
        batch_size_dict = {
            f"{mode}_batch_size_per_gpu": self.base_batch_size * (i + 1) for i, mode in enumerate(SUPPORTED_MODES)
        }

        # Ensure final derived batch size for each mode on GPU is correct
        if torch.cuda.is_available():
            same_batch_size_dict = {"device": "cuda:0", "device_ids": [0]}
            different_batch_sizes_dict = {
                "device": "cuda:0",
                "device_ids": [0],
                "batch_size_per_gpu": None,
                **batch_size_dict,
            }
            self._test_all_batch_sizes(same_batch_size_dict, different_batch_sizes_dict, 1)

            # Ensure final derived batch size for each mode on multiple GPUs is correct
            n_gpu = self.n_gpu
            if n_gpu > 1:
                same_batch_size_dict = {"device": "cuda:0", "device_ids": -1}
                different_batch_sizes_dict = {
                    "device": "cuda:0",
                    "device_ids": -1,
                    "batch_size_per_gpu": None,
                    **batch_size_dict,
                }
                self._test_all_batch_sizes(same_batch_size_dict, different_batch_sizes_dict, n_gpu)

        else:
            raise unittest.SkipTest("GPU(s) not available.")

    def _test_all_batch_sizes(
        self, same_batch_size_dict: Dict, different_batch_sizes_dict: Dict, multiplier: int
    ) -> None:
        """
        Test the configuration of batch size assuming
        (1) common, and (2) different batch sizes for each mode.
        """
        config = self._load_config(same_batch_size_dict)
        self.assertEqual(getattr(config, "train_batch_size"), self.base_batch_size * multiplier)
        self.assertEqual(getattr(config, "eval_batch_size"), self.base_batch_size * multiplier)
        self.assertEqual(getattr(config, "test_batch_size"), self.base_batch_size * multiplier)

        config = self._load_config(different_batch_sizes_dict)
        self.assertEqual(getattr(config, "train_batch_size"), self.base_batch_size * multiplier)
        self.assertEqual(getattr(config, "eval_batch_size"), self.base_batch_size * 2 * multiplier)
        self.assertEqual(getattr(config, "test_batch_size"), self.base_batch_size * 3 * multiplier)

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
