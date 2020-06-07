from __future__ import annotations
import numpy as np
import pandas as pd
import os
import logging
import dill
from itertools import islice
from munch import Munch
from typing import Any, List, Tuple, Dict, Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from pytorch_common import timing
from .additional_configs import BaseDatasetConfig, BaseModelConfig
from .utils import (
    get_model_outputs_only, send_batch_to_device, send_model_to_device,
    send_optimizer_to_device, remove_object, get_checkpoint_name, ModelTracker
)


_string_dict = Dict[str, Any]
_config = Union[_string_dict, Munch]
_loss_or_losses = Union[_Loss, Iterable[_Loss]]
_eval_criterion_or_criteria = Union[Dict[str, Callable], Dict[str, List[Callable]]]
_device = Union[str, torch.device]
_train_result = List[float]
_eval_result = Tuple[List[float], Dict[str, float], torch.Tensor, torch.Tensor]
_test_result = Iterable[torch.Tensor]


@timing
def train_model(
    model: nn.Module,
    config: _config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    loss_criterion_train: _loss_or_losses,
    loss_criterion_eval: _loss_or_losses,
    eval_criteria: _eval_criterion_or_criteria,
    train_logger: ModelTracker,
    val_logger: ModelTracker,
    epochs: Optional[int] = None,
    scheduler: Optional[object] = None,
    early_stopping: Optional[EarlyStopping] = None,
    config_info_dict: Optional[_string_dict] = None,
    start_epoch: Optional[int] = 0,
    decouple_fn_train: Optional[Callable] = None,
    decouple_fn_eval: Optional[Callable] = None
) -> _string_dict:
    """
    Perform the entire model training routine.
      - `epochs` is deliberately not derived directly from config
        so as to be able to change it on the fly without modifying config.
      - `start_epoch` may be provided if a trained checkpoint is loaded
        into the model and training is to be resumed from that point.
      - `decouple_fn_train` and `decouple_fn_eval` are functions which
        take in the batch and return the separated out inputs
        (and targets for training/evaluation).
        They may be specified if this process deviates from the
        default behavior (see `decouple_batch_train`).

    NOTE: Training may be paused at any time with a keyboard interrupt.
          However, please avoid interrupting after an epoch is finished
          and before the next one begins, e.g. during saving a
          checkpoint, as it may cause issues while loading the model.
          Instead pause it during training/evaluation within an epoch.

    :param loss_criterion_train: Training loss criterion
    :param loss_criterion_eval: Evaluation loss criterion
    :param eval_criteria: Dict of evaluation criteria.
                          Keys are names and values are
                          the respective functions.
    :param train_logger: Logger for performance metrics
                         on training set
    :param val_logger: Logger for performance metrics
                       on validation set
    :param config_info_dict: Dict comprising additional information
                             about the config which will be used to
                             generate a unique string for the
                             checkpoint name
    :param start_epoch: May be set to the last trained epoch
                        if training is resumed from an
                        earlier saved checkpoint
    :param decouple_fn_train: Decoupling function to extract
                              inputs and targets from a batch
                              during training
    :param decouple_fn_eval: Decoupling function to extract
                             inputs from a batch during evaluation
    """
    # Provision to override epochs
    # Otherwise derive from config
    if epochs is None:
        epochs = config.epochs

    best_epoch, stop_epoch = 0, start_epoch
    best_checkpoint_file, best_model = "", None
    for epoch in range(1+start_epoch, 1+start_epoch+epochs):
        try:
            # Train epoch
            train_losses = train_epoch(
                model=model,
                dataloader=train_loader,
                device=config.device,
                loss_criterion=loss_criterion_train,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler if config.use_scheduler_after_step else None,
                decouple_fn=decouple_fn_train
            )

            # Evaluate on training set
            _, eval_metrics_train, _, _ = evaluate_epoch(
                model=model,
                dataloader=train_loader,
                device=config.device,
                loss_criterion=loss_criterion_eval,
                eval_criteria=eval_criteria,
                decouple_fn=decouple_fn_eval
            )
            # Add train losses+eval metrics, and log them
            train_logger.add_and_log_metrics(train_losses, eval_metrics_train)

            # Evaluate on val set
            val_losses, eval_metrics_val, _, _ = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                device=config.device,
                loss_criterion=loss_criterion_eval,
                eval_criteria=eval_criteria,
                decouple_fn=decouple_fn_eval
            )
            # Add val losses+eval metrics, and log them
            val_logger.add_and_log_metrics(val_losses, eval_metrics_val)

            # Take scheduler step
            if config.use_scheduler_after_epoch:
                take_scheduler_step(scheduler, np.mean(val_losses))

            # Get early stopping metric
            early_stopping_metric = val_logger.get_early_stopping_metric()

            # Set best epoch
            # Check if current epoch better than previous best based
            # on early stopping (if used) or all epoch history
            if (config.use_early_stopping and early_stopping.is_better(early_stopping_metric))\
                or (not config.use_early_stopping and epoch == val_logger.get_overall_best_epoch()):
                logging.info("Computing best epoch and adding to validation logger...")
                val_logger.set_best_epoch(epoch)
                logging.info("Done.")

                # Replace model checkpoint if required
                if not config.disable_checkpointing:
                    logging.info("Replacing current best model checkpoint...")
                    best_checkpoint_file = save_model(model, optimizer, config,
                                                      train_logger, val_logger, epoch,
                                                      config_info_dict, scheduler)
                    remove_model(config, best_epoch, config_info_dict)
                    best_epoch = epoch
                    logging.info("Done.")

            # Quit training if stopping criterion met
            if config.use_early_stopping and early_stopping.stop(early_stopping_metric):
                stop_epoch = epoch
                logging.info(f"Stopping early after {stop_epoch} epochs.")
                break

            stop_epoch = epoch # Update last epoch trained
        except KeyboardInterrupt: # Option to quit training with keyboard interrupt
            logging.warning("Keyboard Interrupted!")
            stop_epoch = epoch - 1 # Current epoch training incomplete
            break

    # Save the model checkpoints
    if not config.disable_checkpointing:
        logging.info("Dumping model and results...")
        save_model(model, optimizer, config, train_logger,
                   val_logger, stop_epoch, config_info_dict, scheduler)

        # Save current and best models
        save_model(model.copy(), optimizer, config, train_logger, val_logger,
                   stop_epoch, config_info_dict, scheduler, checkpoint_type="model")
        if best_checkpoint_file != "":
            checkpoint = load_model(model.copy(), config, best_checkpoint_file,
                                    optimizer, scheduler)
            best_model = checkpoint["model"]
            optimizer, scheduler = checkpoint["optimizer"], checkpoint["scheduler"]
            checkpoint = None # Free up memory
            save_model(best_model, optimizer, config, train_logger, val_logger,
                       best_epoch, config_info_dict, scheduler, checkpoint_type="model")
        logging.info("Done.")

    return_dict = {
        "model": model,
        "best_model": best_model if best_model is not None else model,
        "train_logger": train_logger,
        "val_logger": val_logger,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "stop_epoch": stop_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint_file": best_checkpoint_file
    }
    return return_dict

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: _device,
    loss_criterion: _loss_or_losses,
    epoch: int,
    optimizer: Optimizer,
    scheduler: Optional[object] = None,
    decouple_fn: Optional[Callable] = None
) -> _train_result:
    """
    Perform one training epoch and return the loss per example
    for each iteration.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch("train", model, dataloader, device,
                             loss_criterion=loss_criterion,
                             epoch=epoch, optimizer=optimizer,
                             scheduler=scheduler, decouple_fn=decouple_fn)

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: _device,
    loss_criterion: _loss_or_losses,
    eval_criteria: _eval_criterion_or_criteria,
    decouple_fn: Optional[Callable] = None
) -> _eval_result:
    """
    Perform one evaluation epoch and return the loss per example
    for each epoch, all eval criteria, raw model outputs, and
    the true targets.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch("eval", model, dataloader, device,
                             loss_criterion=loss_criterion,
                             eval_criteria=eval_criteria,
                             decouple_fn=decouple_fn)

@torch.no_grad()
def get_all_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: _device,
    threshold_prob: Optional[float] = None,
    decouple_fn: Optional[Callable] = None
) -> _test_result:
    """
    Make predictions on entire dataset and return raw outputs
    and optionally class predictions and probabilities if it's
    a classification model.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch("test", model, dataloader, device,
                             threshold_prob=threshold_prob,
                             decouple_fn=decouple_fn)

@timing
def perform_one_epoch(
    phase: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: _device,
    loss_criterion: Optional[_loss_or_losses] = None,
    epoch: Optional[int] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
    eval_criteria: Optional[_eval_criterion_or_criteria] = None,
    threshold_prob: Optional[float] = None,
    decouple_fn: Optional[Callable] = None
) -> Union[_train_result, _eval_result, _test_result]:
    """
    Common loop for one training / evaluation / testing epoch on the entire dataset.
      - For training, returns the loss per example for each iteration.
      - For evaluation, returns the loss per example for each epoch, all eval
        criteria, raw model outputs, and the true targets.
      - For testing, returns raw model outputs, and optionally class predictions
        and probabilities if it's a classification model.

    :param phase: Type of pass to perform over data
                  Choices = "train" | "eval" | "test"
    :param scheduler: Pass this only if it's a scheduler that requires taking a step
                      after each batch iteration (e.g. CyclicLR), otherwise None

    If `phase=="train"`, params `optimizer` and `epoch` must be provided.
    If `phase=="eval"`, param `eval_criteria` must be provided.

    If `phase=="test"`, the dataloader may not have the true labels (by
    definition), and hence, the decoupling function must only return the
    inputs. For the other two phases, they must return the targets as well.

    At a time, only one of training / evaluation / testing will be performed.
    For a given phase, all arguments that pertain to other phases will be ignored.
    """
    ALLOWED_PHASES = ["train", "eval", "test"]

    # Check presence of required arguments
    if phase == "train":
        for param_name, param in zip(["epoch", "optimizer", "loss_criterion"],
                                     [epoch, optimizer, loss_criterion]):
            assert param is not None, f"Param '{param_name}' must not be None for training."
    elif phase == "eval":
        for param_name, param in zip(["eval_criteria", "loss_criterion"],
                                     [eval_criteria, loss_criterion]):
            assert param is not None, f"Param '{param_name}' must not be None for evaluation."
    elif phase != "test":
        raise ValueError(f"Param 'phase' ('{phase}') must be one of {ALLOWED_PHASES}.")

    # Mode for retaining gradients / graph
    MODE = phase == "train"

    # Set decoupling function to extract inputs (and optionally targets) from batch
    if decouple_fn is None:
        decouple_fn = decouple_batch_test if phase == "test" else decouple_batch_train

    # Set model in training/eval mode as required
    model.train(mode=MODE)

    # Get required dataloader params
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Print 50 times in an epoch (or every time, if num_batches < 50)
    batches_to_print = np.unique(np.linspace(0, num_batches, num=50, endpoint=True, dtype=int))

    # Store all required items to be returned
    loss_hist, targets_hist, outputs_hist, preds_hist, probs_hist = [], [], [], [], []

    # Enable gradient computation if training to be performed else disable it.
    # Technically not required if this function is called from other supported
    # functions, e.g. `evaluate_epoch()` (because of decorator), but just being sure.
    with torch.set_grad_enabled(MODE):
        for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
            # Get inputs for testing
            if phase == "test":
                inputs = send_batch_to_device(decouple_fn(batch), device)
            else: # Get inputs and targets for training/evaluation
                inputs, targets = send_batch_to_device(decouple_fn(batch), device)

            # Reset gradients to zero
            if phase == "train":
                optimizer.zero_grad()
                model.zero_grad()

            # Get model outputs
            outputs = get_model_outputs_only(model(inputs))

            # Store items for testing + print progress
            if phase == "test":
                outputs = send_batch_to_device(outputs, "cpu")
                outputs_hist.extend(outputs)

                # Get class predictions and probabilities
                if model.model_type == "classification":
                    preds, probs = model.predict_proba(outputs, threshold_prob)
                    preds_hist.extend(preds)
                    probs_hist.extend(probs)

                # Print progess
                if batch_idx in batches_to_print:
                    logging.info("{}/{} ({:.0f}%) complete.".format(
                        (batch_idx+1) * batch_size, num_examples,
                        100. * (batch_idx+1) / num_batches))

            else: # Perform training / evaluation
                # Compute and store loss
                loss = loss_criterion(outputs, targets)
                loss_value = loss.item()
                loss_hist.append(loss_value)

                # Perform training
                if phase == "train":
                    # Backprop + clip gradients + take scheduler step
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    if scheduler is not None:
                        take_scheduler_step(scheduler, loss_value)

                    # Print progess
                    if batch_idx in batches_to_print:
                        logging.info("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch, (batch_idx+1) * batch_size, num_examples,
                            100. * (batch_idx+1) / num_batches, loss_value))

                else: # Store items for evaluation
                    outputs, targets = send_batch_to_device((outputs, targets), "cpu")
                    outputs_hist.append(outputs)
                    targets_hist.append(targets)

    # Reset gradients back to zero
    if phase == "train":
        optimizer.zero_grad()
        model.zero_grad()

    elif phase == "eval": # Perform evaluation on whole dataset
        outputs_hist = torch.cat(outputs_hist, dim=0)
        targets_hist = torch.cat(targets_hist, dim=0)

        # Compute all evaluation criteria
        eval_metrics = {eval_criterion: eval_fn(outputs_hist, targets_hist) \
                        for eval_criterion, eval_fn in eval_criteria.items()}

    else: # Get outputs, predictions, probabilities
        outputs_hist = torch.stack(outputs_hist, dim=0)
        if model.model_type == "classification":
            preds_hist = torch.stack(preds_hist, dim=0)
            probs_hist = torch.stack(probs_hist, dim=0)

    # Return necessary items
    if phase == "train":
        return loss_hist
    elif phase == "eval":
        return loss_hist, eval_metrics, outputs_hist, targets_hist
    else:
        return outputs_hist, preds_hist, probs_hist

def decouple_batch_train(batch) -> Tuple:
    """
    Separate out batch into inputs and targets
    by assuming they're the first two elements
    in the batch, and return them.
    Used commonly during training/evaluation.

    This is required because often other things
    are also passed in the batch for debugging.
    """
    # Assume first two elements of
    # batch are (inputs, targets)
    inputs, targets = batch[:2]
    return inputs, targets

def decouple_batch_test(batch):
    """
    Extract and return just the inputs
    from a batch assuming it's the first
    element in the batch.
    Used commonly to make test-time predictions.

    This is required because often other things
    are also passed in the batch for debugging.
    """
    # Only inputs are needed for making predictions
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch

def take_scheduler_step(scheduler: object, val_metric: Optional[float] = None) -> None:
    """
    Take a scheduler step.
    Some schedulers, e.g. `ReduceLROnPlateau`, require
    the validation metric to take a step, while (most)
    others don't.
    """
    REQUIRE_VAL_METRIC = ["ReduceLROnPlateau"]

    scheduler_name = scheduler.__class__.__name__
    if scheduler_name in REQUIRE_VAL_METRIC:
        assert val_metric is not None, (f"Param 'val_metric' must be provided "
                                        f"for '{scheduler_name}' scheduler.")
        scheduler.step(val_metric)
    else:
        scheduler.step()

def save_model(
    model: nn.Module,
    optimizer: Optimizer,
    config: _config,
    train_logger: ModelTracker,
    val_logger: ModelTracker,
    epoch: int,
    config_info_dict: Optional[_string_dict] = None,
    scheduler: Optional[object] = None,
    checkpoint_type: Optional[str] = "state"
) -> str:
    """
    Save the checkpoint at a given epoch.
    It can save either:
      - the entire model (copied to CPU).
        This is NOT recommended since it breaks
        if the model code is changed.
      - or just its state dict.
    Additionally, it saves the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config

    :param checkpoint_type: Type of checkpoint to load
                            Choices = "state" | "model"
                            Default = "state"
    :returns name of checkpoint file
    """
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type)

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model, epoch, config_info_dict)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    logging.info(f"Saving {checkpoint_type} checkpoint '{checkpoint_path}'...")

    # Generate appropriate checkpoint dictionary
    checkpoint = generate_checkpoint_dict(optimizer, config, train_logger,
                                          val_logger, epoch, scheduler)

    # Save model in appropriate way
    if checkpoint_type == "state":
        checkpoint["model"] = model.module.state_dict() if hasattr(model, "module") \
                              else model.state_dict()
    else:
        checkpoint["model"] = send_model_to_device(model, "cpu") # Save model on CPU

    # `dill` is used when a model has serialization issues, e.g. that
    # caused by having a lambda function as an attribute of the model.
    # Regular pickling won't work, but it will with dill.
    # Note 1: Avoid using it if possible since it's a little slower.
    # Note 2: It has more robust serialization though.
    # Note 3: When `checkpoint_type="state"`, it should automatically
    #         always work with pickle.
    try:
        torch.save(checkpoint, checkpoint_path)
    except AttributeError:
        torch.save(checkpoint, checkpoint_path, pickle_module=dill)

    logging.info("Done.")
    return checkpoint_file

def generate_checkpoint_dict(
    optimizer: Optimizer,
    config: _config,
    train_logger: ModelTracker,
    val_logger: ModelTracker,
    epoch: int,
    scheduler: Optional[object] = None
) -> Dict[str, Union[Optimizer, object]]:
    """
    Generate a dictionary for storing a checkpoint.
    Helper function for `save_model()`.
    It saves the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    """
    checkpoint = {
        "train_logger": train_logger,
        "val_logger": val_logger,
        "config": config, # Good practice to store config too
        "epoch": epoch,
        "optimizer": optimizer.state_dict()
    }

    # Save scheduler if provided
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    return checkpoint

def load_model(
    model: nn.Module,
    config: _config,
    checkpoint_file: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
    checkpoint_type: Optional[str] = "state"
) -> _string_dict:
    """
    Load the checkpoint at a given epoch.
    It can load either:
      - the entire model
      - or just its state dict into a pre-defined model
        Note: Input model should be pre-defined in this case.
              This routine only updates its state.
    Additionally, it loads the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    Note: Input optimizer and scheduler should be pre-defined
          if their states are to be updated.

    :param model: Must be None if the whole model is to be loaded,
                  or an already created model must be passed if
                  only its state dict is to be updated.
    :param checkpoint_file: Name of the checkpoint present in
                            `config.checkpoint_dir`
    :param checkpoint_type: Type of checkpoint to load
                            Choices = "state" | "model"
                            Default = "state"
    """
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type, checkpoint_file)

    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading {checkpoint_type} checkpoint '{checkpoint_path}'...")

        # See `save_model()` for explanation
        try:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
        except AttributeError:
            checkpoint = torch.load(checkpoint_path, map_location=config.device, pickle_module=dill)

        # Load model in appropriate way
        if checkpoint_type == "state": # Load state dict
            assert model is not None
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])
        else: # Load entire model
            assert model is None
            model = checkpoint["model"]

        # Load train / val loggers
        train_logger = checkpoint["train_logger"]
        val_logger = checkpoint["val_logger"]

        # Load config
        config = checkpoint["config"]

        # Load optimizer and scheduler state dicts
        optimizer, scheduler = load_optimizer_and_scheduler(checkpoint, config.device,
                                                            optimizer, scheduler)

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(os.path.splitext(checkpoint_file)[0].split("-epoch_")[-1])

        # Verify consistency of last epoch trained
        assert epoch_trained == checkpoint["epoch"], \
            (f"Mismatch between epoch specified in checkpoint path ('{epoch_trained}'), "
             f"epoch specified at saving time ('{checkpoint['epoch']}').")

        # Throw warning if model trained for more epochs
        if max(train_logger.epochs) > epoch_trained:
            logging.warning(
                f"The specified epoch was {epoch_trained} but the model was trained for "
                f"{max(train_logger.epochs)} epochs. Ignore this warning if it was intentional."
            )

        # Throw warning if best epoch is different
        if val_logger.best_epoch != epoch_trained:
            logging.warning(
                f"The specified epoch was {epoch_trained} but the best epoch based on validation "
                f"set was {val_logger.best_epoch}. Ignore this warning if it was intentional."
            )

        logging.info("Done.")

    else:
        raise FileNotFoundError(f"No {checkpoint_type} checkpoint found at '{checkpoint_path}'.")

    # Prepare dict to be returned
    return_dict = {
        "model": model,
        "train_logger": train_logger,
        "val_logger": val_logger,
        "config": config,
        "epoch": epoch_trained,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
    return return_dict

def load_optimizer_and_scheduler(
    checkpoint: _string_dict,
    device: _device,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None
) -> Tuple[Optional[Optimizer], Optional[object]]:
    """
    Load the state dict of a given optimizer
    and scheduler, if they're provided.
    Helper function for `load_model()`.
    """
    def load_state_dict(obj, key="optimizer"):
        """
        Properly load state dict of optimizer/scheduler.
        """
        state_dict = checkpoint.get(key)
        if state_dict is not None:
            obj.load_state_dict(state_dict)
        else:
            raise KeyError(
                f"{key} argument expected its state dict in "
                f"the loaded checkpoint but none was found."
            )
        return obj

    # Load optimizer
    if optimizer is not None:
        optimizer = load_state_dict(optimizer, "optimizer")
        optimizer = send_optimizer_to_device(optimizer, device)

    # Load scheduler
    if scheduler is not None:
        scheduler = load_state_dict(scheduler, "scheduler")

    return optimizer, scheduler

def remove_model(
    config: _config,
    epoch: Optional[int],
    config_info_dict: Optional[_string_dict] = None,
    checkpoint_type: Optional[str] = "state"
) -> None:
    """
    Remove a checkpoint/model at a given epoch.
    Used in early stopping if better performance
    is observed at a subsequent epoch.

    :param config_info_dict: Dict comprising additional information
                             about the config which will be used to
                             generate a unique string for the
                             checkpoint name
    :param checkpoint_type: Type of checkpoint to load
                            Choices = "state" | "model"
                            Default = "state"
    """
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type)

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model, epoch, config_info_dict)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logging.info(f"Removing {checkpoint_type} checkpoint '{checkpoint_path}'...")
        remove_object(checkpoint_path)
        logging.info("Done.")

def validate_checkpoint_type(
    checkpoint_type: str,
    checkpoint_file: Optional[str] = None
) -> None:
    """
    Check that the passed `checkpoint_type` is valid and matches that
    obtained from `checkpoint_file`, if provided.
    """
    ALLOWED_CHECKPOINT_TYPES = ["state", "model"]
    assert checkpoint_type in ALLOWED_CHECKPOINT_TYPES, \
        (f"Param 'checkpoint_type' ('{checkpoint_type}') "
         f"must be one of {ALLOWED_CHECKPOINT_TYPES}.")

    # Check that provided checkpoint_type matches that of checkpoint_file
    if checkpoint_file is not None:
        file_checkpoint_type = checkpoint_file.split("-", 3)[1]
        assert file_checkpoint_type == checkpoint_type, \
            (f"The type of checkpoint provided in param "
             f"'checkpoint_type' ('{checkpoint_type}') does "
             f"not match that obtained from the model at "
             f"'{checkpoint_file}' ('{file_checkpoint_type}').")


class EarlyStopping(object):
    """
    Implements early stopping in PyTorch.
    Reference: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
               with a few improvements.
    Common metrics (mse, accuracy, etc.) are ineherently supported, so specifying
    their params is optional.
    """
    SUPPORTED_CRITERIA = ["mse", "accuracy", "precision", "recall", "f1", "auc"]
    SUPPORTED_MODES = {"minimize": 0., "maximize": 1.}
    CRITERIA_MODE_DICT = {
        "mse": "minimize",
        "accuracy": "maximize",
        "precision": "maximize",
        "recall": "maximize",
        "f1": "maximize",
        "auc": "maximize"
    }
    DEFAULT_PARAMS = {"min_delta": 0.2*1e-3, "patience": 5, "best_val_tol": 5e-3}

    def __init__(
        self,
        criterion: Optional[str] = "f1",
        mode: Optional[str] = None,
        min_delta: Optional[float] = None,
        patience: Optional[int] = None,
        best_val: Optional[float] = None,
        best_val_tol: Optional[float] = None
    ):
        """
        :param criterion: name of early stopping criterion
                          If criterion is supported ineherently, you may choose to
                          not provide any of the other params and use the default ones.
        :param mode: whether to "maximize" or "minimize" the `criterion`
        :param min_delta: Minimum difference in metric required to prevent early stopping
        :param patience: No. of epochs (or steps) over which to monitor early stopping
        :param best_val: Best possible value of metric (if any)
        :param best_val_tol: Tolerance when comparing metric to best_val
                             This must be provided if `best_val` is provided
        """
        self.criterion = criterion
        self._init_params(mode=mode, min_delta=min_delta, patience=patience,
                          best_val=best_val, best_val_tol=best_val_tol)
        self._validate_params()
        self.best = None
        self.num_bad_epochs = 0

        if self.patience == 0:
            self.is_better = lambda metric: True
            self.stop = lambda metric: False

    def _init_params(self, **kwargs):
        """
        Initialize all params.
        If it's a supported criterion, specifying parameters is optional.
        If not, all of them must be provided (with the exception of
        `best_val` and `best_val_tol`).
        """
        # Supported criterion
        if self.criterion in self.SUPPORTED_CRITERIA:
            mode = self.CRITERIA_MODE_DICT[self.criterion]
            best_val = self.SUPPORTED_MODES[mode]
            params = {**self.DEFAULT_PARAMS, "mode": mode, "best_val": best_val}

            # Keep specified params and fall back to default for others
            kwargs = {k: v if v is not None else params[k] for k, v in kwargs.items()}

        else:
            # Non-default params must be provided for unsupported criteria
            for k, v in kwargs.items():
                if k not in self.DEFAULT_PARAMS.keys():
                    assert v is not None, (f"The only criteria currently supported by "
                                           f"default are {self.SUPPORTED_CRITERIA}, "
                                           f"and hence param '{k}' is required.")

        # Finally set values and validate them
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _validate_params(self):
        """
        Check validity of mode of optimization.
        """
        supported_modes = self.SUPPORTED_MODES.keys()
        if self.mode not in supported_modes:
            raise ValueError(f"Param 'mode' ('{self.mode}') must be one of {supported_modes}.")
        if self.best_val is not None and self.best_val_tol is None:
            raise ValueError("Param 'best_val_tol' must be provided if 'best_val' is provided.")

    def is_better(self, metric: float) -> bool:
        """
        Check if the provided `metric` is the best one so far.
        """
        if self.best is None: # After first step
            return True
        if self.mode == "minimize":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def stop(self, metric: float) -> bool:
        """
        Check if early stopping criterion met.
        """
        if self.best is None: # First step
            self.best = metric
            return False

        if np.isnan(metric):
            return True

        if self.is_better(metric): # Reset patience counter if better
            self.num_bad_epochs = 0
            self.best = metric
        else: # Otherwise increment counter by 1
            self.num_bad_epochs += 1

        # Check if already reached max value
        if self.best_val is not None and np.abs(self.best_val-metric) < self.best_val_tol:
            return True

        # Check if patience counter reached
        if self.num_bad_epochs >= self.patience:
            return True

        return False
