import unittest
import numpy as np
import itertools
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
        """
        Load pytorch_common config and override it
        with given default parameters.
        Also get all objects required for training,
        like model, dataloaders, loggers, optimizer, etc.
        """
        cls._load_config()

    @classmethod
    def tearDownClass(cls):
        """
        Delete all directories created during config initialization.
        """
        for directory in ["transientdir", "packagedir", "misc_data_dir"]:
            utils.remove_dir(cls.config[directory], force=True)

    def test_early_stopping(self):
        """
        Test different early stopping criteria.
        """
        for eval_criterion in EVAL_CRITERIA:
            early_stopping = train_utils.EarlyStopping(eval_criterion)
        self._test_error(train_utils.EarlyStopping, "dummy_criterion")

    def test_all_models(self, **kwargs):
        """
        Test all models for all compatible
        datasets by:
          - ensuring that saving/loading model works
          - performing the whole training routine
          - getting all predictions at test time
        """
        all_kwargs = self._get_all_combination_kwargs()

        for model_type in all_kwargs.keys():
            self._load_config(self.default_config_dict) # Reload default config
            self.config.model_type = model_type
            loss_criterion = "cross-entropy" if model_type == "classification" else "mse"
            eval_criterion = "accuracy" if model_type == "classification" else "mse"
            for dataset_kwargs, model_kwargs in all_kwargs[model_type]:
                kwargs = {"dataset_kwargs": dataset_kwargs, "model_kwargs": model_kwargs}
                self._test_saving_loading_models(loss_criterion, eval_criterion, **model_kwargs)
                self._test_train_model(loss_criterion, eval_criterion, **kwargs)
                self._test_get_all_predictions(loss_criterion, eval_criterion, **kwargs)

    def _get_all_combination_kwargs(self):
        """
        Generate a list of kwargs for all compatible
        combinations of datasets and models.
        """
        # Define dataset and model hyperparams
        size, in_dim, h_dim, out_dim, num_layers, num_classes = 5, 4, 4, 1, 2, 2

        # Define classification kwargs
        classification_dataset_kwargs = [
            {"dataset_name": "multi_class_dataset", "size": size, "dim": in_dim,
             "num_classes": num_classes},
            {"dataset_name": "multi_label_dataset", "size": size, "dim": in_dim,
             "num_classes": num_classes, "multilabel_reduction": "mean"}
        ]

        classification_model_kwargs = [
            {"model_name": "single_layer_classifier", "in_dim": in_dim, "num_classes": num_classes},
            {"model_name": "multi_layer_classifier", "in_dim": in_dim, "num_classes": num_classes,
             "h_dim": h_dim, "num_layers": num_layers}
        ]

        classification_kwargs = list(itertools.product(classification_dataset_kwargs,
                                                       classification_model_kwargs))

        # Define regression kwargs
        regression_dataset_kwargs = [
            {"dataset_name": "regression_dataset", "size": size,
             "in_dim": in_dim, "out_dim": out_dim}
        ]

        regression_model_kwargs = [
            {"model_name": "single_layer_regressor", "in_dim": in_dim, "out_dim": out_dim},
            {"model_name": "multi_layer_regressor", "in_dim": in_dim, "out_dim": out_dim,
             "h_dim": h_dim, "num_layers": num_layers}
        ]

        regression_kwargs = list(itertools.product(regression_dataset_kwargs,
                                                   regression_model_kwargs))

        all_kwargs = {"classification": classification_kwargs, "regression": regression_kwargs}
        return all_kwargs

    def _test_saving_loading_models(self, loss_criterion, eval_criterion, **model_kwargs):
        """
        Test saving and loading of models.
        """
        # Get required training objects
        model = self._get_model(**model_kwargs)
        optimizer = self._get_optimizer(model)
        train_logger, val_logger = self._get_loggers(loss_criterion, eval_criterion)

        # Ensure that original model state dict matches the
        # one obtained from saving and then loading the model
        model_state_dict_orig = model.state_dict()
        checkpoint_file = train_utils.save_model(model, optimizer, self.config,
                                                 train_logger, val_logger, 1)
        return_dict = train_utils.load_model(model, self.config,
                                             checkpoint_file, optimizer)
        self.assertTrue(utils.compare_model_state_dicts(model_state_dict_orig,
                        return_dict["model"].state_dict()))

    def _test_train_model(self, loss_criterion, eval_criterion, **kwargs):
        """
        Test the whole training routine of a model.
        """
        # Get all training objects
        return_dict = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Train model
        train_utils.train_model(return_dict["model"], self.config, return_dict["train_loader"],
                                return_dict["val_loader"], return_dict["optimizer"],
                                return_dict["loss_criterion_train"],
                                return_dict["loss_criterion_test"], return_dict["eval_criteria"],
                                return_dict["train_logger"], return_dict["val_logger"],
                                self.config.epochs)

    def _test_get_all_predictions(self, loss_criterion, eval_criterion, **kwargs):
        """
        Test the routine of obtaining predictions from a model.
        """
        # Get all training objects
        return_dict = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Get all predictions
        outputs_val, preds_val, probs_val = \
            train_utils.get_all_predictions(return_dict["model"], return_dict["val_loader"],
                                            self.config.device)
        train_utils.get_all_predictions(return_dict["model"], return_dict["val_loader"],
                                        self.config.device, threshold_prob=0.8)

        # Ensure shape of predictions is correct
        self.assertEqual(len(outputs_val), len(return_dict["val_loader"].dataset))
        if return_dict["model"].model_type == "classification":
            for results in [preds_val, probs_val]:
                self.assertEqual(len(results), len(return_dict["val_loader"].dataset))

    def _test_error(self, func, args, error=AssertionError):
        """
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args`.
        """
        with self.assertRaises(error):
            func(args)

    @classmethod
    def _load_config(cls, dictionary=None):
        """
        Load the default pytorch_common config
        after overriding it with `dictionary`.
        """
        dictionary = cls._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        cls.config = config

    @classmethod
    def _get_merged_dict(cls, dictionary=None):
        """
        Override default config with
        `dictionary` if provided.
        """
        cls.default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cls.default_config_dict = {
            "load_pytorch_common_config": True,
            "transientdir": "dummy_transientdir",
            "packagedir": "dummy_package_dir",
            "misc_data_dir": "dummy_misc_data_dir",
            "device": cls.default_device,
            "epochs": 1
        }

        if dictionary is None:
            return cls.default_config_dict
        return {**cls.default_config_dict, **dictionary}

    def _get_training_objects(self, loss_criterion, eval_criterion, **kwargs):
        """
        Get all objects required for training, like
        model, dataloaders, loggers, optimizer, etc.
        """
        dataset_kwargs, model_kwargs = kwargs["dataset_kwargs"], kwargs["model_kwargs"]

        multilabel_reduction = dataset_kwargs.pop("multilabel_reduction", None)

        # Get training/validation dataloaders
        train_loader, val_loader = self._get_dataloaders(**dataset_kwargs)

        # Get model
        model = self._get_model(**model_kwargs)

        # Get optimizer
        optimizer = self._get_optimizer(model)

        # Get training/validation loggers
        train_logger, val_logger = self._get_loggers(loss_criterion, eval_criterion)

        # Get training/testing loss and eval criteria
        if multilabel_reduction is not None:
            self.config.classification_type = "multilabel"
            self.config.loss_kwargs["multilabel_reduction"] = multilabel_reduction
            self.config.eval_criteria_kwargs["multilabel_reduction"] = multilabel_reduction
        loss_criterion_train, loss_criterion_test, eval_criteria = \
            get_loss_eval_criteria(self.config, reduction="mean")

        training_objects = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "model": model,
            "train_logger": train_logger,
            "val_logger": val_logger,
            "optimizer": optimizer,
            "loss_criterion_train": loss_criterion_train,
            "loss_criterion_test": loss_criterion_test,
            "eval_criteria": eval_criteria
        }
        return training_objects

    def _get_dataloaders(self, **kwargs):
        """
        Get training and validation dataloaders.
        """
        dataset_name = kwargs.pop("dataset_name")
        dataset_config = BaseDatasetConfig(kwargs)
        dataset = create_dataset(dataset_name, dataset_config)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=kwargs["size"])
        val_loader = DataLoader(dataset, shuffle=False, batch_size=kwargs["size"])

        return train_loader, val_loader

    def _get_model(self, **kwargs):
        """
        Get model.
        """
        model_name = kwargs.pop("model_name")
        model_config = BaseModelConfig(kwargs)
        model = create_model(model_name, model_config)
        model = utils.send_model_to_device(model, self.config.device, self.config.device_ids)
        return model

    def _get_optimizer(self, model):
        """
        Get optimizer.
        """
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer

    def _get_loggers(self, loss_criterion, eval_criterion):
        """
        Get training and validation
        loggers and feed in some dummy
        logs to emulate training.
        """
        self.config.loss_criterion = loss_criterion
        self.config.eval_criteria = [eval_criterion]
        self.config.early_stopping_criterion = eval_criterion
        train_logger, val_logger = utils.get_model_performance_trackers(self.config)
        train_logger.add_metrics(np.random.randn(10), {eval_criterion: np.random.randn(10)})
        val_logger.add_metrics(np.random.randn(self.config.epochs),
                               {eval_criterion: np.random.randn(self.config.epochs)})
        val_logger.set_best_epoch(1)

        return train_logger, val_logger


if __name__ == "__main__":
    unittest.main()
