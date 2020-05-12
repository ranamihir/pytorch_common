import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from pytorch_common.config import load_pytorch_common_config, set_pytorch_config
from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.datasets import DummyMultiClassDataset
from pytorch_common.models import create_model
from pytorch_common.metrics import EVAL_CRITERIA, get_loss_eval_criteria
from pytorch_common import train_utils, utils


class TestTrainUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cls.config = {
            'load_pytorch_common_config': True,
            'transientdir': 'dummy_transientdir',
            'packagedir': 'dummy_package_dir',
            'misc_data_dir': 'dummy_misc_data_dir',
            'device': cls.default_device,
            'epochs': 1
        }
        cls.config = cls._load_config(cls.config)

        cls._get_training_objects()

    @classmethod
    def tearDownClass(cls):
        for directory in ['transientdir', 'packagedir', 'misc_data_dir']:
            utils.remove_dir(cls.config[directory], force=True)

    def test_early_stopping(self):
        for eval_criterion in EVAL_CRITERIA:
            early_stopping = train_utils.EarlyStopping(eval_criterion)
        self._test_error(train_utils.EarlyStopping, 'dummy_criterion')

    def test_saving_loading_models(self):
        model_state_dict_orig = self.model.state_dict()
        checkpoint_file = train_utils.save_model(self.model, self.optimizer, self.config,
                                                 self.train_logger, self.val_logger, 1)
        return_dict = train_utils.load_model(self.model, self.config,
                                             checkpoint_file, self.optimizer)
        self.assertTrue(utils.compare_model_state_dicts(model_state_dict_orig,
                        return_dict['model'].state_dict()))

    def test_train_model(self):
        train_utils.train_model(self.model, self.config, self.train_loader, self.val_loader,
                                self.optimizer, self.loss_criterion_train, self.loss_criterion_test,
                                self.eval_criteria, self.train_logger, self.val_logger,
                                self.config.epochs)

    def test_get_all_predictions(self):
        train_utils.get_all_predictions(self.model, self.val_loader, self.config.device)
        train_utils.get_all_predictions(self.model, self.val_loader, self.config.device,
                                        threshold_prob=0.8)

    def _test_error(self, func, args, error=AssertionError):
        with self.assertRaises(error):
            func(args)

    @classmethod
    def _load_config(cls, dictionary=None):
        dictionary = cls._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        return config

    @classmethod
    def _get_merged_dict(cls, dictionary=None):
        if dictionary is None:
            return cls.config.copy()
        return {**cls.config, **dictionary}

    @classmethod
    def _get_training_objects(cls):
        size, dim, num_classes = 5, 1, 1
        cls.train_loader = DataLoader(DummyMultiClassDataset(size=size, dim=dim,
                                      num_classes=num_classes), shuffle=True, batch_size=size)
        cls.val_loader = DataLoader(DummyMultiClassDataset(size=size, dim=dim,
                                    num_classes=num_classes), shuffle=False, batch_size=size)

        model_config = BaseModelConfig({'in_dim': dim, 'num_classes': num_classes})
        cls.model = create_model('single_layer_classifier', model_config)
        cls.model = utils.send_model_to_device(cls.model, cls.config.device, cls.config.device_ids)

        cls.train_logger, cls.val_logger = utils.get_model_performance_trackers(cls.config)
        cls.train_logger.add_metrics(np.random.randn(10), {'accuracy': np.random.randn(10)})
        cls.val_logger.add_metrics(np.random.randn(cls.config.epochs),
                                   {'accuracy': np.random.randn(cls.config.epochs)})
        cls.val_logger.set_best_epoch(1)

        cls.optimizer = SGD(cls.model.parameters(), lr=1e-3)

        cls.loss_criterion_train, cls.loss_criterion_test, cls.eval_criteria = \
            get_loss_eval_criteria(cls.config, reduction='mean')


if __name__ == '__main__':
    unittest.main()
