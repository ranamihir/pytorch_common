import unittest
import torch

from pytorch_common.config import load_pytorch_common_config, set_pytorch_config
from pytorch_common import utils


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        cls.config = {
            'load_pytorch_common_config': True,
            'transientdir': 'dummy_transientdir',
            'packagedir': 'dummy_package_dir',
            'misc_data_dir': 'dummy_misc_data_dir',
            'device': cls.default_device
        }

        cls.n_gpu = torch.cuda.device_count()

    @classmethod
    def tearDownClass(cls):
        for directory in ['transientdir', 'packagedir', 'misc_data_dir']:
            utils.remove_dir(cls.config[directory], force=True)

    def test_load_pytorch_common_config(self):
        config = self._load_config({'classification_type': 'multiclass'})
        self.assertEqual(config.classification_type, 'multiclass')
        self.assertEqual(config.device, self.default_device)
        self.assertFalse(config.assert_gpu)

        dictionary = {'load_pytorch_common_config': False}
        self.assertEqual(dict(self._load_config(dictionary)), self._get_merged_dict(dictionary))

    def test_set_loss_and_eval_criteria(self):
        self._test_config_error({'model_type': 'regression', 'loss_criterion': 'accuracy'})
        self._test_config_error({'model_type': 'classification', 'eval_criteria': ['accuracy'],
                                 'use_early_stopping': False, 'early_stopping_criterion': 'f1'})
        self._test_config_error({'model_type': 'classification', 'eval_criteria': ['accuracy'],
                                 'use_early_stopping': True, 'early_stopping_criterion': 'f1'})
        self._test_config_error({'model_type': 'classification', 'classification_type': 'dummy'})

        for cls_type in ['binary', 'multiclass', 'multilabel']:
            self._test_config_error({'model_type': 'regression', 'classification_type': cls_type})

        self._load_config({'model_type': 'classification', 'classification_type': 'binary',
                           'loss_criterion': 'focal-loss'})
        for cls_type in ['multiclass', 'multilabel']:
            self._test_config_error({'model_type': 'classification',
                                     'classification_type': cls_type,
                                     'loss_criterion': 'focal-loss'})

    def test_check_and_set_devices(self):
        self._test_config_error({'device': 'cpu', 'device_ids': [0,1]})
        self.assertEqual(self._load_config({'device': 'cpu'}).n_gpu, 0)

        if torch.cuda.is_available():
            config = self._load_config({'device': 'cuda:0', 'device_ids': [1,0]})
            self.assertEqual(config.device_ids, [0,1])

            if self.n_gpu > 1:
                config = self._load_config({'device': 'cuda:0', 'device_ids': -1})
                self.assertEqual(config.device_ids, list(range(self.n_gpu)))

    def test_set_batch_size(self):
        self.assertEqual(self._load_config({'batch_size': 8}).batch_size, 8)
        self.assertEqual(self._load_config({'device': 'cpu', 'batch_size': 8}).batch_size, 8)
        self._test_config_error({'batch_size': 8, 'batch_size_per_gpu': 8}, error=ValueError)

        if torch.cuda.is_available():
            config = self._load_config({'device': 'cuda:0', 'device_ids': [0],
                                        'batch_size_per_gpu': 8})
            self.assertEqual(config.batch_size, 8)

            if self.n_gpu > 1:
                config = self._load_config({'device': 'cuda:0', 'device_ids': -1,
                                            'batch_size_per_gpu': 8})
                self.assertEqual(config.batch_size, 8*self.n_gpu)

    def _test_config_error(self, dictionary, error=AssertionError):
        with self.assertRaises(error):
            self._load_config(dictionary)

    def _load_config(self, dictionary=None):
        dictionary = self._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        return config

    def _get_merged_dict(self, dictionary=None):
        if dictionary is None:
            return self.config.copy()
        return {**self.config, **dictionary}


if __name__ == '__main__':
    unittest.main()
