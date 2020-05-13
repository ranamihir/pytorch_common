import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

from pytorch_common.config import load_pytorch_common_config, set_pytorch_config
from pytorch_common.additional_configs import BaseDatasetConfig, BaseModelConfig
from pytorch_common.datasets import create_dataset
from pytorch_common.models import create_model
from pytorch_common.metrics import EVAL_CRITERIA, get_loss_eval_criteria
from pytorch_common import train_utils, utils


class TestTrainUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Load pytorch_common config and override it
        with given default parameters.
        Also get all objects required for training,
        like model, dataloaders, loggers, optimizer, etc.
        '''
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

        # Define default dataset and model hyperparams
        cls.default_dataset_kwargs = {'dataset_name': 'multi_class_dataset', 'size': 5,
                                      'dim': 8, 'num_classes': 2}
        cls.default_model_kwargs = {'model_name': 'single_layer_classifier', 'in_dim': 8,
                                    'num_classes': 2}

    @classmethod
    def tearDownClass(cls):
        '''
        Delete all directories created during config initialization.
        '''
        for directory in ['transientdir', 'packagedir', 'misc_data_dir']:
            utils.remove_dir(cls.config[directory], force=True)

    def test_early_stopping(self):
        '''
        Test different early stopping criteria.
        '''
        for eval_criterion in EVAL_CRITERIA:
            early_stopping = train_utils.EarlyStopping(eval_criterion)
        self._test_error(train_utils.EarlyStopping, 'dummy_criterion')

    def test_saving_loading_models(self):
        '''
        Test saving and loading of models.
        '''
        # Get required training objects
        model = self._get_model(**self.default_model_kwargs)
        optimizer = self._get_optimizer(model)
        train_logger, val_logger = self._get_loggers(eval_criterion='accuracy')

        # Ensure that original model state dict matches the
        # one obtained from saving and then loading the model
        model_state_dict_orig = model.state_dict()
        checkpoint_file = train_utils.save_model(model, optimizer, self.config,
                                                 train_logger, val_logger, 1)
        return_dict = train_utils.load_model(model, self.config,
                                             checkpoint_file, optimizer)
        self.assertTrue(utils.compare_model_state_dicts(model_state_dict_orig,
                        return_dict['model'].state_dict()))

    def test_train_model(self):
        '''
        Test the whole training routine of a model.
        '''
        # Get all training objects
        kwargs = {'dataset_kwargs': self.default_dataset_kwargs,
                  'model_kwargs': self.default_model_kwargs}
        return_dict = self._get_training_objects(eval_criterion='accuracy', **kwargs)

        # Train model
        train_utils.train_model(return_dict['model'], self.config, return_dict['train_loader'],
                                return_dict['val_loader'], return_dict['optimizer'],
                                return_dict['loss_criterion_train'],
                                return_dict['loss_criterion_test'], return_dict['eval_criteria'],
                                return_dict['train_logger'], return_dict['val_logger'],
                                self.config.epochs)

    def test_get_all_predictions(self):
        '''
        Test the routine of obtaining predictions from a model.
        '''
        # Get all training objects
        kwargs = {'dataset_kwargs': self.default_dataset_kwargs,
                  'model_kwargs': self.default_model_kwargs}
        return_dict = self._get_training_objects(eval_criterion='accuracy', **kwargs)

        # Get all predictions
        outputs_val, preds_val, probs_val = \
            train_utils.get_all_predictions(return_dict['model'], return_dict['val_loader'],
                                            self.config.device)
        train_utils.get_all_predictions(return_dict['model'], return_dict['val_loader'],
                                        self.config.device, threshold_prob=0.8)

        # Ensure shape of predictions is correct
        for results in [outputs_val, preds_val, probs_val]:
            self.assertEqual(len(results), len(return_dict['val_loader'].dataset))

    def _test_error(self, func, args, error=AssertionError):
        '''
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args`.
        '''
        with self.assertRaises(error):
            func(args)

    @classmethod
    def _load_config(cls, dictionary=None):
        '''
        Load the default pytorch_common config
        after overriding it with `dictionary`.
        '''
        dictionary = cls._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        return config

    @classmethod
    def _get_merged_dict(cls, dictionary=None):
        '''
        Override default config with
        `dictionary` if provided.
        '''
        if dictionary is None:
            return cls.config.copy()
        return {**cls.config, **dictionary}

    @classmethod
    def _get_training_objects(cls, eval_criterion, **kwargs):
        '''
        Get all objects required for training, like
        model, dataloaders, loggers, optimizer, etc.
        '''
        dataset_kwargs, model_kwargs = kwargs['dataset_kwargs'], kwargs['model_kwargs']

        # Get training/validation dataloaders
        train_loader, val_loader = cls._get_dataloaders(**dataset_kwargs)

        # Get model
        model = cls._get_model(**model_kwargs)

        # Get optimizer
        optimizer = cls._get_optimizer(model)

        # Get training/validation loggers
        train_logger, val_logger = cls._get_loggers(eval_criterion=eval_criterion)

        # Get training/testing loss and eval criteria
        loss_criterion_train, loss_criterion_test, eval_criteria = \
            get_loss_eval_criteria(cls.config, reduction='mean')

        training_objects = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'model': model,
            'train_logger': train_logger,
            'val_logger': val_logger,
            'optimizer': optimizer,
            'loss_criterion_train': loss_criterion_train,
            'loss_criterion_test': loss_criterion_test,
            'eval_criteria': eval_criteria
        }
        return training_objects

    @classmethod
    def _get_dataloaders(cls, **kwargs):
        '''
        Get training and validation dataloaders
        '''
        dataset_name = kwargs.pop('dataset_name')
        dataset_config = BaseDatasetConfig(kwargs)
        dataset = create_dataset(dataset_name, dataset_config)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=kwargs['size'])
        val_loader = DataLoader(dataset, shuffle=False, batch_size=kwargs['size'])

        return train_loader, val_loader

    @classmethod
    def _get_model(cls, **kwargs):
        '''
        Get model
        '''
        model_name = kwargs.pop('model_name')
        model_config = BaseModelConfig(kwargs)
        model = create_model(model_name, model_config)
        model = utils.send_model_to_device(model, cls.config.device, cls.config.device_ids)
        return model

    @classmethod
    def _get_optimizer(cls, model):
        '''
        Get optimizer
        '''
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer

    @classmethod
    def _get_loggers(cls, **kwargs):
        '''
        Get training/validation loggers and feed
        in some dummy logs to emulate training
        '''
        metric = kwargs['eval_criterion']
        train_logger, val_logger = utils.get_model_performance_trackers(cls.config)
        train_logger.add_metrics(np.random.randn(10), {metric: np.random.randn(10)})
        val_logger.add_metrics(np.random.randn(cls.config.epochs),
                                   {metric: np.random.randn(cls.config.epochs)})
        val_logger.set_best_epoch(1)

        return train_logger, val_logger


if __name__ == '__main__':
    unittest.main()
