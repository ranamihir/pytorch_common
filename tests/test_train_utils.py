import itertools
import unittest

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_common import train_utils, utils
from pytorch_common.additional_configs import BaseDatasetConfig, BaseModelConfig
from pytorch_common.config import Config, load_pytorch_common_config, set_pytorch_config
from pytorch_common.datasets import create_dataset
from pytorch_common.metrics import EvalCriteria, get_loss_eval_criteria
from pytorch_common.models import create_model
from pytorch_common.types import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    _StringDict,
)


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
        Delete artifact directory created during config initialization.
        """
        utils.remove_dir(cls.config["artifact_dir"], force=True)

    def test_early_stopping(self):
        """
        Test different early stopping criteria.
        """
        for eval_criterion in EvalCriteria().names:
            early_stopping = train_utils.EarlyStopping(eval_criterion)
        self._test_error(train_utils.EarlyStopping, criterion="dummy_criterion")

    def test_all_models(self, **kwargs):
        """
        Test all models for all compatible
        datasets by:
          - ensuring that saving / loading model works
          - performing the whole training routine
          - getting all predictions at test time
        """
        all_kwargs = self._get_all_combination_kwargs()

        for model_type in all_kwargs.keys():
            self._load_config(self.default_config_dict)  # Reload default config
            self.config.model_type = model_type
            loss_criterion = "cross-entropy" if model_type == "classification" else "mse"
            eval_criterion = "accuracy" if model_type == "classification" else "mse"
            for dataset_kwargs, model_kwargs in all_kwargs[model_type]:
                kwargs = {"dataset_kwargs": dataset_kwargs, "model_kwargs": model_kwargs}
                self._test_partial_saving_loading_model(loss_criterion, eval_criterion, **model_kwargs)
                self._test_full_saving_loading_model(loss_criterion, eval_criterion, **model_kwargs)
                self._test_train_model(loss_criterion, eval_criterion, **kwargs)
                self._test_train_return_keys(loss_criterion, eval_criterion, **kwargs)
                self._test_eval_return_keys(loss_criterion, eval_criterion, **kwargs)
                self._test_prediction_return_keys(loss_criterion, eval_criterion, **kwargs)
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
            {
                "dataset_name": "multi_class_dataset",
                "size": size,
                "dim": in_dim,
                "num_classes": num_classes,
            },
            {
                "dataset_name": "multi_label_dataset",
                "size": size,
                "dim": in_dim,
                "num_classes": num_classes,
                "multilabel_reduction": "mean",
            },
        ]

        classification_model_kwargs = [
            {"model_name": "single_layer_classifier", "in_dim": in_dim, "num_classes": num_classes},
            {
                "model_name": "multi_layer_classifier",
                "in_dim": in_dim,
                "num_classes": num_classes,
                "h_dim": h_dim,
                "num_layers": num_layers,
            },
        ]

        classification_kwargs = list(itertools.product(classification_dataset_kwargs, classification_model_kwargs))

        # Define regression kwargs
        regression_dataset_kwargs = [
            {
                "dataset_name": "regression_dataset",
                "size": size,
                "in_dim": in_dim,
                "out_dim": out_dim,
            }
        ]

        regression_model_kwargs = [
            {"model_name": "single_layer_regressor", "in_dim": in_dim, "out_dim": out_dim},
            {
                "model_name": "multi_layer_regressor",
                "in_dim": in_dim,
                "out_dim": out_dim,
                "h_dim": h_dim,
                "num_layers": num_layers,
            },
        ]

        regression_kwargs = list(itertools.product(regression_dataset_kwargs, regression_model_kwargs))

        all_kwargs = {"classification": classification_kwargs, "regression": regression_kwargs}
        return all_kwargs

    def _test_partial_saving_loading_model(self, loss_criterion: str, eval_criterion: str, **model_kwargs) -> None:
        """
        Test saving and loading of just the model.
        """
        # Get required training objects
        model = self._get_model(**model_kwargs)

        # Ensure that original model state dict matches the
        # one obtained from saving and then loading the model
        model_state_dict_orig = model.state_dict()
        checkpoint_file = train_utils.save_model(model, self.config, 1)
        checkpoint = train_utils.load_model(model, self.config, checkpoint_file)
        self.assertTrue(utils.compare_model_state_dicts(model_state_dict_orig, checkpoint["model"].state_dict()))

    def _test_full_saving_loading_model(self, loss_criterion: str, eval_criterion: str, **model_kwargs) -> None:
        """
        Test saving and loading of model alongwith
        other objects, e.g. optimizer, scheduler, etc.
        """
        # Get required training objects
        model = self._get_model(**model_kwargs)
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        train_logger, val_logger = self._get_loggers(loss_criterion, eval_criterion)

        # Ensure that original model state dict matches the
        # one obtained from saving and then loading the model
        model_state_dict_orig = model.state_dict()
        checkpoint_file = train_utils.save_model(model, self.config, 1, train_logger, val_logger, optimizer, scheduler)
        checkpoint = train_utils.load_model(model, self.config, checkpoint_file, optimizer, scheduler)
        self.assertTrue(utils.compare_model_state_dicts(model_state_dict_orig, checkpoint["model"].state_dict()))

    def _test_train_model(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the whole training routine of a model.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Train model
        train_utils.train_model(
            training_objects["model"],
            self.config,
            training_objects["train_loader"],
            training_objects["optimizer"],
            training_objects["loss_criterion_train"],
            training_objects["loss_criterion_test"],
            training_objects["eval_criteria"],
            training_objects["train_logger"],
            training_objects["val_loader"],
            training_objects["val_logger"],
            self.config.epochs,
            training_objects["scheduler"],
        )

        # Rerain model without evaluation
        train_utils.train_model(
            training_objects["model"],
            self.config,
            training_objects["train_loader"],
            training_objects["optimizer"],
            training_objects["loss_criterion_train"],
            training_objects["loss_criterion_test"],
            training_objects["eval_criteria"],
            training_objects["train_logger"],
            val_loader=None,
            val_logger=None,
            epochs=self.config.epochs,
            scheduler=training_objects["scheduler"],
        )

    def _test_train_return_keys(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the keys returned by `train_utils.train_epoch()`.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Training epoch
        return_dict = train_utils.train_epoch(
            model=training_objects["model"],
            dataloader=training_objects["train_loader"],
            device=self.config.device,
            loss_criterion=training_objects["loss_criterion_train"],
            epoch=0,
            optimizer=training_objects["optimizer"],
            epochs=self.config.epochs,
        )
        self.assertEqual(sorted(list(return_dict)), ["losses"])

    def _test_eval_return_keys(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the keys returned by `train_utils.evaluate_epoch()`.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        true_return_keys = ["losses", "eval_metrics"]

        # Evaluation epoch
        for return_keys in [[], None, ["outputs"]]:
            return_dict = train_utils.evaluate_epoch(
                model=training_objects["model"],
                dataloader=training_objects["train_loader"],
                device=self.config.device,
                loss_criterion=training_objects["loss_criterion_train"],
                eval_criteria=training_objects["eval_criteria"],
                return_keys=return_keys,
            )
            if return_keys is None:
                additional_keys = ["outputs", "targets"]
            else:
                additional_keys = return_keys
            self.assertEqual(sorted(list(return_dict)), sorted(true_return_keys + additional_keys))

    def _test_prediction_return_keys(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the keys returned by `train_utils.get_all_predictions()`.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Get all predictions
        for return_keys in [None, ["probs"]]:
            return_dict = train_utils.get_all_predictions(
                model=training_objects["model"],
                dataloader=training_objects["train_loader"],
                device=self.config.device,
                return_keys=return_keys,
            )
            if return_keys is None:
                additional_keys = ["outputs", "probs", "preds"]
            else:
                additional_keys = return_keys
            self.assertEqual(sorted(list(return_dict)), sorted(additional_keys))

        # Get all predictions
        self._test_error(
            train_utils.get_all_predictions,
            ValueError,
            model=training_objects["model"],
            dataloader=training_objects["train_loader"],
            device=self.config.device,
            return_keys=[],
        )

    def _test_get_all_predictions(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the routine of obtaining predictions from a model.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Get all predictions
        return_dict = train_utils.get_all_predictions(
            training_objects["model"], training_objects["val_loader"], self.config.device
        )
        train_utils.get_all_predictions(
            training_objects["model"], training_objects["val_loader"], self.config.device, threshold_prob=0.8
        )

        # Ensure shape of predictions is correct
        self.assertEqual(len(return_dict["outputs"]), len(training_objects["val_loader"].dataset))
        if training_objects["model"].model_type == "classification":
            for results in [return_dict["preds"], return_dict["probs"]]:
                self.assertEqual(len(results), len(training_objects["val_loader"].dataset))

    def _test_error(self, func: Callable[[Any], None], error=AssertionError, *args, **kwargs) -> None:
        """
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args` and `kwargs`.
        """
        with self.assertRaises(error):
            func(*args, **kwargs)

    @classmethod
    def _load_config(cls, dictionary: Optional[Dict] = None) -> Config:
        """
        Load the default pytorch_common config
        after overriding it with `dictionary`.
        """
        dictionary = cls._get_merged_dict(dictionary)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        cls.config = config

    @classmethod
    def _get_merged_dict(cls, dictionary: Optional[Dict] = None) -> Dict:
        """
        Override default config with
        `dictionary` if provided.
        """
        cls.default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cls.default_config_dict = {
            "load_pytorch_common_config": True,
            "artifact_dir": "dummy_artifact_dir",
            "device": cls.default_device,
            "train_batch_size_per_gpu": 5,
            "eval_batch_size_per_gpu": 5,
            "test_batch_size_per_gpu": 5,
            "epochs": 1,
        }

        if dictionary is None:
            return cls.default_config_dict
        return {**cls.default_config_dict, **dictionary}

    def _get_training_objects(self, loss_criterion: str, eval_criterion: str, **kwargs) -> _StringDict:
        """
        Get all objects required for training, like
        model, dataloaders, loggers, optimizer, etc.
        """
        dataset_kwargs, model_kwargs = kwargs["dataset_kwargs"], kwargs["model_kwargs"]

        multilabel_reduction = dataset_kwargs.pop("multilabel_reduction", None)

        # Get training / validation dataloaders
        train_loader, val_loader = self._get_dataloaders(**dataset_kwargs)

        # Get model
        model = self._get_model(**model_kwargs)

        # Get optimizer
        optimizer = self._get_optimizer(model)

        # Get scheduler
        scheduler = self._get_scheduler(optimizer)

        # Get training / validation loggers
        train_logger, val_logger = self._get_loggers(loss_criterion, eval_criterion)

        # Get training / testing loss and eval criteria
        if multilabel_reduction is not None:
            self.config.classification_type = "multilabel"
            self.config.loss_kwargs["multilabel_reduction"] = multilabel_reduction
            self.config.eval_criteria_kwargs["multilabel_reduction"] = multilabel_reduction
        loss_criterion_train, loss_criterion_test, eval_criteria = get_loss_eval_criteria(self.config)

        training_objects = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "model": model,
            "train_logger": train_logger,
            "val_logger": val_logger,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "loss_criterion_train": loss_criterion_train,
            "loss_criterion_test": loss_criterion_test,
            "eval_criteria": eval_criteria,
        }
        return training_objects

    def _get_dataloaders(self, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        Get training and validation dataloaders.
        """
        dataset_name = kwargs.pop("dataset_name")
        dataset_config = BaseDatasetConfig(kwargs)
        dataset = create_dataset(dataset_name, dataset_config)
        train_loader = DataLoader(dataset, shuffle=True, batch_size=self.config.train_batch_size_per_gpu)
        val_loader = DataLoader(dataset, shuffle=False, batch_size=self.config.eval_batch_size_per_gpu)
        return train_loader, val_loader

    def _get_model(self, **kwargs) -> nn.Module:
        """
        Get model.
        """
        model_name = kwargs.pop("model_name")
        model_config = BaseModelConfig(kwargs)
        model = create_model(model_name, model_config)
        model = utils.send_model_to_device(model, self.config.device, self.config.device_ids)
        return model

    def _get_optimizer(self, model: nn.Module) -> Optimizer:
        """
        Get optimizer.
        """
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer

    def _get_scheduler(self, optimizer: Optimizer) -> object:
        """
        Get scheduler.
        """
        scheduler = ReduceLROnPlateau(optimizer)
        return scheduler

    def _get_loggers(self, loss_criterion: str, eval_criterion: str) -> Tuple[utils.ModelTracker, utils.ModelTracker]:
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
        val_logger.add_metrics(
            np.random.randn(self.config.epochs),
            {eval_criterion: np.random.randn(self.config.epochs)},
        )
        val_logger.set_best_epoch(1)

        return train_logger, val_logger


if __name__ == "__main__":
    unittest.main()
