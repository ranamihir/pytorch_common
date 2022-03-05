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
from pytorch_common.metrics import LOSS_REDUCTIONS, EvalCriteria, get_loss_eval_criteria
from pytorch_common.models import create_model
from pytorch_common.types import Any, Callable, Tuple, _StringDict


def is_multilabel_dataset(dataset_kwargs) -> bool:
    """
    Check if it's a multi-label dataset.
    """
    return "multilabel_reduction" in dataset_kwargs


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

    def test_all_models(self):
        """
        Test all models for all compatible
        datasets by:
          - ensuring that saving / loading model works
          - performing the whole training routine
          - getting all predictions at test time
        """
        all_kwargs = self._get_all_combination_kwargs()

        for model_type in all_kwargs:
            self._load_config()  # Reload default config
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
                if not is_multilabel_dataset(dataset_kwargs):  # Sample weighting not supported for multilabel
                    self._test_sample_weighting_reduction(loss_criterion, eval_criterion, **kwargs)

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
            model=training_objects["model"],
            config=self.config,
            train_loader=training_objects["train_loader"],
            optimizer=training_objects["optimizer"],
            loss_criterion_train=training_objects["loss_criterion_train"],
            loss_criterion_eval=training_objects["loss_criterion_test"],
            eval_criteria=training_objects["eval_criteria"],
            train_logger=training_objects["train_logger"],
            val_loader=training_objects["val_loader"],
            val_logger=training_objects["val_logger"],
            epochs=self.config.epochs,
            scheduler=training_objects["scheduler"],
            sample_weighting_train=self.config.sample_weighting_train,
            sample_weighting_eval=self.config.sample_weighting_eval,
        )

        # Retrain model without evaluation
        train_utils.train_model(
            model=training_objects["model"],
            config=self.config,
            train_loader=training_objects["train_loader"],
            optimizer=training_objects["optimizer"],
            loss_criterion_train=training_objects["loss_criterion_train"],
            loss_criterion_eval=training_objects["loss_criterion_test"],
            eval_criteria=training_objects["eval_criteria"],
            train_logger=training_objects["train_logger"],
            val_loader=None,
            val_logger=None,
            epochs=self.config.epochs,
            scheduler=training_objects["scheduler"],
            sample_weighting_train=self.config.sample_weighting_train,
            sample_weighting_eval=self.config.sample_weighting_eval,
        )

    def _test_train_return_keys(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the keys returned by `train_utils.train_epoch()`.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Get loss reduction parameter if using sample weighting
        loss_reduction_train = (
            self.config.loss_kwargs.get("reduction_train", "mean") if self.config.sample_weighting_train else None
        )

        # Training epoch
        return_dict = train_utils.train_epoch(
            model=training_objects["model"],
            dataloader=training_objects["train_loader"],
            device=self.config.device,
            loss_criterion=training_objects["loss_criterion_train"],
            epoch=0,
            optimizer=training_objects["optimizer"],
            epochs=self.config.epochs,
            sample_weighting=self.config.sample_weighting_train,
            loss_reduction=loss_reduction_train,
        )
        self.assertEqual(sorted(list(return_dict)), ["losses"])

    def _test_eval_return_keys(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test the keys returned by `train_utils.evaluate_epoch()`.
        """
        # Get all training objects
        training_objects = self._get_training_objects(loss_criterion, eval_criterion, **kwargs)

        # Get loss reduction parameter if using sample weighting
        loss_reduction_eval = (
            self.config.loss_kwargs.get("reduction_val", "mean") if self.config.sample_weighting_eval else None
        )

        true_return_keys = ["losses", "eval_metrics"]

        # Evaluation epoch
        for return_keys in [[], None, ["outputs"]]:
            return_dict = train_utils.evaluate_epoch(
                model=training_objects["model"],
                dataloader=training_objects["train_loader"],
                device=self.config.device,
                loss_criterion=training_objects["loss_criterion_train"],
                eval_criteria=training_objects["eval_criteria"],
                sample_weighting=self.config.sample_weighting_eval,
                loss_reduction=loss_reduction_eval,
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

    def _test_sample_weighting_reduction(self, loss_criterion: str, eval_criterion: str, **kwargs) -> None:
        """
        Test training all models with sample weighting.
        """
        orig_config, orig_kwargs = self.config.copy(), kwargs.copy()

        # Ensure loss reduction is correctly specified if using sample weighting
        for mode in ["train", "val"]:
            kwargs = orig_kwargs.copy()
            mode_ = "eval" if mode == "val" else mode
            kwargs[f"sample_weighting_{mode_}"] = True
            kwargs["loss_kwargs"] = kwargs.get("loss_kwargs", {})
            for loss_reduction in LOSS_REDUCTIONS:  # Test valid configurations
                kwargs["loss_kwargs"][f"reduction_{mode}"] = loss_reduction
                self._load_config(**kwargs)  # Reload config with params
                self._test_train_model(loss_criterion, eval_criterion, **kwargs)
            for loss_reduction in ["dummy", None]:  # Test invalid configurations
                kwargs["loss_kwargs"][f"reduction_{mode}"] = loss_reduction
                self._load_config(**kwargs)  # Reload config with params
                self._test_error(
                    self._test_train_model,
                    AssertionError,
                    loss_criterion=loss_criterion,
                    eval_criterion=eval_criterion,
                    **kwargs,
                )

        self._load_config(**dict(orig_config))  # Reload original config

    def _test_error(self, func: Callable[[Any], None], error=AssertionError, *args, **kwargs) -> None:
        """
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args` and `kwargs`.
        """
        with self.assertRaises(error):
            func(*args, **kwargs)

    @classmethod
    def _load_config(cls, **kwargs) -> Config:
        """
        Load the default pytorch_common config
        after overriding it with `kwargs`.
        """
        dictionary = cls._get_merged_dict(**kwargs)
        config = load_pytorch_common_config(dictionary)
        set_pytorch_config(config)
        cls.config = config

    @classmethod
    def _get_merged_dict(cls, **kwargs) -> _StringDict:
        """
        Override default config with
        `kwargs` if provided.
        """
        cls.default_device = "cuda:0" if torch.cuda.is_available() else "cpu"

        cls.default_config_dict = {
            "load_pytorch_common_config": True,
            "artifact_dir": "dummy_artifact_dir",
            "device": cls.default_device,
            "train_batch_size_per_gpu": 5,
            "eval_batch_size_per_gpu": 5,
            "test_batch_size_per_gpu": 5,
            "sample_weighting_train": False,
            "sample_weighting_eval": False,
            "epochs": 1,
        }

        if kwargs is None:
            return cls.default_config_dict.copy()
        return {**cls.default_config_dict, **kwargs}

    def _get_training_objects(self, loss_criterion: str, eval_criterion: str, **kwargs) -> _StringDict:
        """
        Get all objects required for training, like
        model, dataloaders, loggers, optimizer, etc.
        """
        dataset_kwargs, model_kwargs = kwargs["dataset_kwargs"], kwargs["model_kwargs"]

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
        if is_multilabel_dataset(dataset_kwargs):
            self.config.classification_type = "multilabel"
            self.config.loss_kwargs["multilabel_reduction"] = dataset_kwargs["multilabel_reduction"]
            self.config.eval_criteria_kwargs["multilabel_reduction"] = dataset_kwargs["multilabel_reduction"]
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
        kwargs.pop("multilabel_reduction", None)
        dataset_name = kwargs.pop("dataset_name")
        dataset_config = BaseDatasetConfig(kwargs)
        dataset = create_dataset(dataset_name, dataset_config)

        # Add weights for sample weighting
        # Inbuilt datasets directly pass everything in the `.data` attribute to `__getitem__()`
        # `"weight"` will automatically be the third column
        dataset.data["weight"] = [torch.tensor(i, dtype=torch.float32) for i in range(len(dataset.data))]

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
