from __future__ import annotations

import os
import re
from collections import OrderedDict
from itertools import islice

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_common import timing

from .metrics import EvalCriteria
from .types import *
from .utils import (
    ModelTracker,
    get_checkpoint_name,
    get_file_path,
    get_model_outputs_only,
    remove_object,
    send_batch_to_device,
    send_model_to_device,
    send_optimizer_to_device,
    setup_logging,
)

logger = setup_logging(__name__)


@timing
def train_model(
    model: nn.Module,
    config: _Config,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_criterion_train: _Loss,
    loss_criterion_eval: _Loss,
    eval_criteria: _EvalCriterionOrCriteria,
    train_logger: ModelTracker,
    val_loader: Optional[DataLoader] = None,
    val_logger: Optional[ModelTracker] = None,
    epochs: Optional[int] = None,
    scheduler: Optional[object] = None,
    early_stopping: Optional[EarlyStopping] = None,
    config_info_dict: Optional[_StringDict] = None,
    checkpoint_file: Optional[str] = None,
    sample_weighting_train: Optional[bool] = False,
    sample_weighting_eval: Optional[bool] = False,
    decouple_fn_train: Optional[_DecoupleFnTrain] = None,
    decouple_fn_eval: Optional[_DecoupleFnTrain] = None,
) -> _StringDict:
    """
    Perform the entire model training routine.
      - `epochs` is deliberately not derived directly from config
        so as to be able to change it on the fly without modifying config.
      - `checkpoint_file` may be provided if a trained checkpoint is to be
        loaded into the model and training is to be resumed from that point.
      - `decouple_fn_train` and `decouple_fn_eval` are functions which
        take in the batch and return the separated out inputs
        (and targets for training/evaluation).
        They may be specified if this process deviates from the default
        behavior (see `decouple_batch_train() for more details`).

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
                       NOTE: Set to None for no evaluation. Useful
                             for retraining from scratch on all data.
                             Must also not provide param `val_loader`
                             in that case.
    :param config_info_dict: Dict comprising additional information
                             about the config which will be used to
                             generate a unique string for the
                             checkpoint name
    :param checkpoint_file: Name of trained checkpoint file
                            if training is to be resumed from
                            that point
    :param sample_weighting_train: Set to True if sample weights are
                                   to be used during training
                                   See `perform_one_epoch()` for more details.
    :param sample_weighting_eval: Set to True if sample weights are
                                  to be used during evaluation
                                  NOTE: Most use cases don't need sample
                                        weights for evaluation. Make sure
                                        you have a strong argument for
                                        using this parameter.
                                  See `perform_one_epoch()` for more details.
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

    # Either both `val_loader` and `val_logger` must be provided or none (for no evaluation)
    assert not ((val_loader is None) ^ (val_logger is None))
    do_evaluation = val_loader is not None

    # Throw error if ReduceLROnPlateau scheduler is used
    if (
        config.use_scheduler_after_epoch or config.use_scheduler_after_step
    ) and scheduler.__class__.__name__ == "ReduceLROnPlateau":
        raise ValueError("Scheduler `ReduceLROnPlateau` is currently not supported.")

    # Load trained model if required
    start_epoch, best_epoch = 0, 0
    if checkpoint_file is not None:
        logger.info("Loading previously saved checkpoint for resuming training...")
        checkpoint = load_model(
            model=model,
            config=config,
            checkpoint_file=checkpoint_file,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        model = checkpoint["model"]
        train_logger = checkpoint["train_logger"]
        if do_evaluation:
            val_logger = checkpoint["val_logger"]
            best_epoch = val_logger.best_epoch
        optimizer, scheduler = checkpoint["optimizer"], checkpoint["scheduler"]
        start_epoch = checkpoint["epoch"]
        del checkpoint  # Free up memory
        logger.info("Done.")

    stop_epoch = start_epoch
    best_checkpoint_file, last_checkpoint_file = "", ""
    best_model: Optional[nn.Module] = None
    for epoch in range(1 + start_epoch, 1 + start_epoch + epochs):
        try:
            # Train epoch
            train_train_result = train_epoch(
                model=model,
                dataloader=train_loader,
                device=config.device,
                loss_criterion=loss_criterion_train,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler if config.use_scheduler_after_step else None,
                epochs=epochs,
                sample_weighting=sample_weighting_train,
                decouple_fn=decouple_fn_train,
            )

            # Evaluate on training set
            train_eval_result = evaluate_epoch(
                model=model,
                dataloader=train_loader,
                device=config.device,
                loss_criterion=loss_criterion_eval,
                eval_criteria=eval_criteria,
                sample_weighting=sample_weighting_eval,
                decouple_fn=decouple_fn_eval,
                return_keys=[],
            )
            # Add train losses+eval metrics, and log them
            train_logger.add_and_log_metrics(train_train_result["losses"], train_eval_result["eval_metrics"])

            # Take scheduler step
            if config.use_scheduler_after_epoch:
                scheduler.step()

            # Evaluate on val set
            if do_evaluation:
                val_result = evaluate_epoch(
                    model=model,
                    dataloader=val_loader,
                    device=config.device,
                    loss_criterion=loss_criterion_eval,
                    eval_criteria=eval_criteria,
                    sample_weighting=sample_weighting_eval,
                    decouple_fn=decouple_fn_eval,
                    return_keys=[],
                )
                # Add val losses+eval metrics, and log them
                val_logger.add_and_log_metrics(val_result["losses"], val_result["eval_metrics"])

                # Get early stopping metric
                early_stopping_metric = val_logger.get_early_stopping_metric()

                # Set best epoch
                # Check if current epoch better than previous best based
                # on early stopping (if used) or all epoch history
                if (config.use_early_stopping and early_stopping.is_better(early_stopping_metric)) or (
                    not config.use_early_stopping and epoch == val_logger.get_overall_best_epoch()
                ):
                    logger.info("Computing best epoch and adding to validation logger...")
                    val_logger.set_best_epoch(epoch)
                    logger.info("Done.")

                    # Replace model checkpoint if required
                    if not config.disable_checkpointing:
                        logger.info("Replacing current best model checkpoint...")
                        best_checkpoint_file = save_model(
                            model,
                            config,
                            epoch,
                            train_logger,
                            val_logger,
                            optimizer,
                            scheduler,
                            config_info_dict,
                        )
                        remove_model(config, best_epoch, config_info_dict)
                        logger.info("Done.")

                    best_epoch = epoch  # Update best epoch

                # Quit training if stopping criterion met
                if config.use_early_stopping and early_stopping.stop(early_stopping_metric):
                    stop_epoch = epoch
                    logger.info(f"Stopping early after {stop_epoch} epochs.")
                    break

            stop_epoch = epoch  # Update last epoch trained
        except KeyboardInterrupt:  # Option to quit training with keyboard interrupt
            logger.warning("Keyboard Interrupted!")
            stop_epoch = epoch - 1  # Current epoch training incomplete
            break

    # Save the model checkpoints
    if not config.disable_checkpointing:
        logger.info("Dumping model and results...")
        last_checkpoint_file = save_model(
            model,
            config,
            stop_epoch,
            train_logger,
            val_logger,
            optimizer,
            scheduler,
            config_info_dict,
        )

        # Save current and best models
        save_model(
            model.copy(),
            config,
            stop_epoch,
            train_logger,
            val_logger,
            optimizer,
            scheduler,
            config_info_dict,
            checkpoint_type="model",
        )

        if do_evaluation and best_checkpoint_file != "":
            checkpoint = load_model(model.copy(), config, best_checkpoint_file, optimizer, scheduler)
            best_model = checkpoint["model"]
            optimizer, scheduler = checkpoint["optimizer"], checkpoint["scheduler"]
            del checkpoint  # Free up memory
            save_model(
                best_model,
                config,
                best_epoch,
                train_logger,
                val_logger,
                optimizer,
                scheduler,
                config_info_dict,
                checkpoint_type="model",
            )

        logger.info("Done.")

    return_dict = {
        "model": model,
        "best_model": best_model if (do_evaluation and best_model is not None) else model,
        "train_logger": train_logger,
        "val_logger": val_logger,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "stop_epoch": stop_epoch,
        "best_epoch": best_epoch if do_evaluation else stop_epoch,
        "best_checkpoint_file": best_checkpoint_file if do_evaluation else last_checkpoint_file,
    }

    return return_dict


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: _Device,
    loss_criterion: _Loss,
    epoch: int,
    optimizer: Optimizer,
    scheduler: Optional[object] = None,
    epochs: Optional[int] = None,
    sample_weighting: Optional[bool] = False,
    decouple_fn: Optional[_DecoupleFnTrain] = None,
) -> _StringDict:
    """
    Perform one training epoch and return the loss per example
    for each iteration.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch(
        phase="train",
        model=model,
        dataloader=dataloader,
        device=device,
        loss_criterion=loss_criterion,
        epoch=epoch,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        sample_weighting=sample_weighting,
        decouple_fn=decouple_fn,
    )


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: _Device,
    loss_criterion: _Loss,
    eval_criteria: _EvalCriterionOrCriteria,
    sample_weighting: Optional[bool] = False,
    decouple_fn: Optional[_DecoupleFnTrain] = None,
    return_keys: Optional[List[str]] = None,
) -> _StringDict:
    """
    Perform one evaluation epoch and return the loss per example
    for each epoch, all eval criteria, raw model outputs, and
    the true targets.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch(
        phase="eval",
        model=model,
        dataloader=dataloader,
        device=device,
        loss_criterion=loss_criterion,
        eval_criteria=eval_criteria,
        sample_weighting=sample_weighting,
        decouple_fn=decouple_fn,
        return_keys=return_keys,
    )


@torch.no_grad()
def get_all_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: _Device,
    threshold_prob: Optional[float] = None,
    decouple_fn: Optional[_DecoupleFnTest] = None,
    return_keys: Optional[List[str]] = None,
) -> _StringDict:
    """
    Make predictions on entire dataset and return raw outputs
    and optionally class predictions and probabilities if it's
    a classification model.
    See `perform_one_epoch()` for more details.
    """
    return perform_one_epoch(
        phase="test",
        model=model,
        dataloader=dataloader,
        device=device,
        threshold_prob=threshold_prob,
        decouple_fn=decouple_fn,
        return_keys=return_keys,
    )


@timing
def perform_one_epoch(
    phase: str,
    model: nn.Module,
    dataloader: DataLoader,
    device: _Device,
    loss_criterion: Optional[_Loss] = None,
    epoch: Optional[int] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
    eval_criteria: Optional[_EvalCriterionOrCriteria] = None,
    threshold_prob: Optional[float] = None,
    epochs: Optional[int] = None,
    sample_weighting: Optional[bool] = False,
    decouple_fn: Optional[_DecoupleFn] = None,
    return_keys: Optional[List[str]] = None,
) -> _StringDict:
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
                      after each batch step/iteration (e.g. CyclicLR), otherwise None
    :param sample_weighting: Whether sample weighting is enabled or not.
                             NOTE: To use this feature, you must provide the weights
                                   in the dataloader decoupling function. See
                                   `decouple_batch_train()` for more details.
    :param return_keys: Additional objects to return in the return dictionary.
                        - For `phase="train"`, this argument will be ignored and
                          the losses will always be returned.
                        - For `phase="eval"`, the losses and evaluation metrics
                          will always be returned. Additionally, you may specify
                          any of `["outputs", "targets"]` in this argument.
                        - For `phase="test"`, you may specify any of
                          `["outputs", "probs", "preds"]` in this argument,
                          otherwise an empty dictionary will be returned.
                        If None, all applicable additional keys will be
                        returned by default.
                        If an empty list, no additional keys will be returned.

    If `phase=="train"`, params `optimizer` and `epoch` must be provided.
    If `phase=="eval"`, param `eval_criteria` must be provided.

    If `phase=="test"`, the dataloader may not have the true labels (by
    definition), and hence, the decoupling function must only return the
    inputs. For the other two phases, they must return the targets as well.

    At a time, only one of training / evaluation / testing will be performed.
    For a given phase, all arguments that pertain to other phases will be ignored.
    """

    def _get_return_key_dict() -> _StringDict:
        """
        Ensure all return keys are expected
        and return a boolean dictionary of
        all keys for the current phase.
        """
        # Set values for mandatory keys to True
        _return_keys = {"losses": True, "eval_metrics": True}

        if return_keys is not None:
            if IS_TRAINING:
                logger.warning(
                    f"You have specified param `return_keys` ({return_keys}) in training phase, but it "
                    f"can only be provided in evaluation or testing phase. This argument will be ignored."
                )
            else:
                _return_keys.update({k: False for k in ALLOWED_RETURN_KEYS[phase]})
                for k in return_keys:
                    if k not in ALLOWED_RETURN_KEYS[phase]:
                        raise ValueError(
                            f"Param 'return_keys' must only comprise keys from {ALLOWED_RETURN_KEYS[phase]}. Got '{k}'."
                        )
                    _return_keys[k] = True

                if (phase == "test") and not len(return_keys):
                    raise ValueError(
                        "At least one key must be specified in param `return_keys` for making predictions."
                    )
        elif not IS_TRAINING:
            _return_keys.update({k: True for k in ALLOWED_RETURN_KEYS[phase]})

        return _return_keys

    def _drop_unnecessary_keys(return_dict: _StringDict, all_keys: List[str], return_keys: _StringDict) -> _StringDict:
        """
        Retain only the keys in `return_dict`
        that are present in `all_keys` and have
        the value in `return_keys` as True.
        """
        keys_to_keep = set(all_keys).intersection(set([k for k, v in return_keys.items() if v]))
        for k in list(return_dict):  # Need `list()` to keep set of keys unchanged after deleting below
            if k not in keys_to_keep:
                del return_dict[k]
        return return_dict

    ALLOWED_PHASES = ["train", "eval", "test"]
    ALLOWED_RETURN_KEYS = {"eval": ["outputs", "targets"], "test": ["outputs", "probs", "preds"]}
    IS_TRAINING = phase == "train"  # Mode for retaining gradients / graph

    # Check presence of required arguments
    if IS_TRAINING:
        for param_name, param in zip(["epoch", "optimizer", "loss_criterion"], [epoch, optimizer, loss_criterion]):
            assert param is not None, f"Param '{param_name}' must not be None for training."
    elif phase == "eval":
        for param_name, param in zip(["eval_criteria", "loss_criterion"], [eval_criteria, loss_criterion]):
            assert param is not None, f"Param '{param_name}' must not be None for evaluation."
    elif phase != "test":
        raise ValueError(f"Param 'phase' ('{phase}') must be one of {ALLOWED_PHASES}.")

    # Get bool dict of return keys
    _return_keys = _get_return_key_dict()

    # Set decoupling function to extract inputs (and optionally targets and sample weights) from batch
    if decouple_fn is None:
        decouple_fn = decouple_batch_test if phase == "test" else decouple_batch_train

    epochs_str = ""
    if epochs is not None:
        assert IS_TRAINING, f"Param `epochs` ({epochs}) can only be provided in training phase."
        epochs_str = f"/{epochs}"

    # Set model in training/eval mode as required
    model.train(mode=IS_TRAINING)

    # Get required dataloader params
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Print 50 times in an epoch (or every time, if num_batches < 50)
    batches_to_print = np.unique(np.linspace(0, num_batches, num=50, endpoint=True, dtype=int))

    # Store all required items to be returned
    return_dict = {"losses": [], "targets": [], "outputs": [], "probs": [], "preds": []}

    # Enable gradient computation if training to be performed else disable it.
    # Technically not required if this function is called from other supported
    # functions, e.g. `evaluate_epoch()` (because of decorator), but just being sure.
    with torch.set_grad_enabled(IS_TRAINING):
        for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
            torch.cuda.empty_cache()  # Often goes OOM without this

            # Get inputs for testing
            if phase == "test":
                inputs = send_batch_to_device(decouple_fn(batch), device)
            else:  # Get inputs, targets, and optionally sample weights for training/evaluation
                batch = send_batch_to_device(decouple_fn(batch, sample_weighting=sample_weighting), device)
                if sample_weighting:
                    inputs, targets, sample_weights = batch
                else:
                    inputs, targets = batch
            del batch  # Free up memory

            # Reset gradients to zero
            if IS_TRAINING:
                optimizer.zero_grad()
                model.zero_grad()

            # Get model outputs
            outputs = get_model_outputs_only(model(inputs))
            del inputs  # Free up memory

            # Store variables for logging
            num_examples_complete = min((batch_idx + 1) * batch_size, num_examples)
            percent_batches_complete = 100.0 * (batch_idx + 1) / num_batches

            # Store items for testing + print progress
            if phase == "test":
                outputs = send_batch_to_device(outputs, "cpu")
                if _return_keys["outputs"]:
                    return_dict["outputs"].extend(outputs)

                # Get class predictions and probabilities
                if (model.model_type == "classification") and (_return_keys["probs"] or _return_keys["preds"]):
                    classification_result = model.predict_proba(
                        outputs, threshold_prob, return_probs=_return_keys["probs"], return_preds=_return_keys["preds"]
                    )
                    for k in ["probs", "preds"]:
                        if _return_keys[k]:
                            return_dict[k].extend(classification_result[k])

                # Print progess
                if batch_idx in batches_to_print:
                    logger.info(f"{num_examples_complete}/{num_examples} ({percent_batches_complete:.0f}%) complete.")

            else:  # Perform training / evaluation
                # Compute and store loss
                loss = loss_criterion(outputs, targets)
                if sample_weighting:
                    loss = (loss * sample_weights / sample_weights.sum()).sum()
                loss_value = loss.item()
                return_dict["losses"].append(loss_value)

                # Perform training
                if IS_TRAINING:
                    del outputs, targets  # Free up memory

                    # Backprop + clip gradients + take scheduler step
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    del loss  # Free up memory

                    # Print progess
                    if batch_idx in batches_to_print:
                        logger.info(
                            f"Train Epoch: {epoch}{epochs_str} [{num_examples_complete}/{num_examples} "
                            f"({percent_batches_complete:.0f}%)]\tLoss: {loss_value:.6f}"
                        )

                else:  # Store items for evaluation
                    outputs, targets = send_batch_to_device((outputs, targets), "cpu")
                    return_dict["outputs"].append(outputs)
                    return_dict["targets"].append(targets)

    # Reset gradients back to zero
    if IS_TRAINING:
        optimizer.zero_grad()
        model.zero_grad()

    elif phase == "eval":  # Perform evaluation on whole dataset
        for k in ["outputs", "targets"]:
            return_dict[k] = torch.cat(return_dict[k], dim=0)

        # Compute all evaluation criteria
        eval_metrics = eval_criteria(return_dict["outputs"], return_dict["targets"])

    else:  # Get outputs, predictions, probabilities
        stack = lambda history: torch.stack(history, dim=0)
        if _return_keys["outputs"]:
            return_dict["outputs"] = stack(return_dict["outputs"])
        if model.model_type == "classification":
            for k in ["probs", "preds"]:
                if _return_keys[k]:
                    return_dict[k] = stack(return_dict[k])

    # Return necessary items
    if IS_TRAINING:
        _drop_unnecessary_keys(return_dict, ["losses"], _return_keys)
    elif phase == "eval":
        _drop_unnecessary_keys(return_dict, ["losses"] + ALLOWED_RETURN_KEYS["eval"], _return_keys)
        if _return_keys["eval_metrics"]:
            return_dict["eval_metrics"] = eval_metrics
    else:
        _drop_unnecessary_keys(return_dict, ALLOWED_RETURN_KEYS["test"], _return_keys)

    return return_dict


def decouple_batch_train(batch: _Batch, sample_weighting: Optional[bool] = False) -> Tuple[_Batch]:
    """
    Separate out batch into inputs and targets
    by assuming they're the first two elements
    in the batch.
    If sample weights are to be used, they
    will assume the third index.

    Used commonly during training/evaluation.

    This is required because often other things
    are also passed in the batch for debugging.
    """
    # Assume first two elements of batch are (inputs, targets)
    inputs, targets = batch[:2]

    # Third is sample weights if required
    if sample_weighting:
        sample_weights = batch[2]
        return inputs, targets, sample_weights
    return inputs, targets


def decouple_batch_test(batch: _Batch) -> _Batch:
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


def save_model(
    model: nn.Module,
    config: _Config,
    epoch: int,
    train_logger: Optional[ModelTracker] = None,
    val_logger: Optional[ModelTracker] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
    config_info_dict: Optional[_StringDict] = None,
    checkpoint_type: Optional[str] = "state",
) -> str:
    """
    Save the checkpoint at a given epoch.
    It can save either:
      - the entire model (copied to CPU).
        This is NOT recommended since it breaks
        if the model code is changed.
      - or just its state dict.
    Additionally, it saves the following variables:
        - Current training config
        - Epoch number
        - History of train and validation losses
          and eval metrics so far (if provided)
        - Optimizer and scheduler state dicts (if provided)

    :param checkpoint_type: Type of checkpoint to load
                            Choices = "state" | "model"
                            Default = "state"
    :returns name of checkpoint file
    """
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type)

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model_name, epoch, config_info_dict)
    checkpoint_path = get_file_path(config.checkpoint_dir, checkpoint_file)
    logger.info(f"Saving {checkpoint_type} checkpoint '{checkpoint_path}'...")

    # Generate appropriate checkpoint dictionary
    checkpoint = generate_checkpoint_dict(config, epoch, train_logger, val_logger, optimizer, scheduler)

    # Save model in appropriate way
    if checkpoint_type == "state":
        checkpoint["model"] = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    else:
        checkpoint["model"] = send_model_to_device(model, "cpu")  # Save model on CPU

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

    logger.info("Done.")
    return checkpoint_file


def generate_checkpoint_dict(
    config: _Config,
    epoch: int,
    train_logger: Optional[ModelTracker] = None,
    val_logger: Optional[ModelTracker] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
) -> Dict[str, Union[_Config, int, ModelTracker, OrderedDict[str, _TensorOrTensors]]]:
    """
    Generate a dictionary for storing a checkpoint.
    Helper function for `save_model()`.
    It saves the following variables:
        - Current training config
        - Epoch number
        - History of train and validation losses
          and eval metrics so far (if provided)
        - Optimizer and scheduler state dicts (if provided)
    """
    checkpoint = {"config": config, "epoch": epoch}  # Good practice to store config too

    # Save items if provided
    for name, obj in zip(
        ("train_logger", "val_logger", "optimizer", "scheduler"),
        (train_logger, val_logger, optimizer, scheduler),
    ):
        if obj is not None:
            checkpoint[name] = obj if name in ["train_logger", "val_logger"] else obj.state_dict()

    return checkpoint


def load_model(
    model: nn.Module,
    config: _Config,
    checkpoint_file: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
) -> _StringDict:
    """
    Load the checkpoint at a given epoch.
    It can load either:
      - the entire model
      - or just its state dict into a pre-defined model
        Note: Input model should be pre-defined in this case.
              This routine only updates its state.
    Additionally, it loads the following variables:
        - Current training config
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Optimizer and scheduler (if provided) state dicts

    Note: Input optimizer and scheduler should be pre-defined
          if their states are to be updated.

    :param model: Must be None if the whole model is to be loaded,
                  or an already created model must be passed if
                  only its state dict is to be updated.
    :param checkpoint_file: Name of the checkpoint present in
                            `config.checkpoint_dir`
    """
    # Extract checkpoint type from the file name provided
    checkpoint_type = get_checkpoint_type_from_file(checkpoint_file)

    checkpoint_path = get_file_path(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logger.info(f"Loading {checkpoint_type} checkpoint '{checkpoint_path}'...")

        # See `save_model()` for explanation
        try:
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
        except AttributeError:
            checkpoint = torch.load(checkpoint_path, map_location=config.device, pickle_module=dill)

        # Load model in appropriate way
        if checkpoint_type == "state":  # Load state dict
            assert model is not None
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint["model"])
        else:  # Load entire model
            assert model is None
            model = checkpoint["model"]

        # Load config
        config = checkpoint["config"]

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(re.search(r"-epoch_(?P<epoch>[0-9]+)", checkpoint_file).group("epoch"))

        # Verify consistency of last epoch trained
        if epoch_trained != checkpoint["epoch"]:
            logger.warning(
                f"Mismatch between epoch specified in checkpoint path ({epoch_trained}) "
                f"and epoch specified at saving time ({checkpoint['epoch']})."
            )

        # Load train / val loggers if provided
        train_logger, val_logger = checkpoint.get("train_logger"), checkpoint.get("val_logger")

        # Throw warning if model trained for more epochs
        if train_logger is not None and max(train_logger.epochs) > epoch_trained:
            logger.warning(
                f"The specified epoch was {epoch_trained} but the model was trained for "
                f"{max(train_logger.epochs)} epochs. Ignore this warning if it was intentional."
            )

        # Throw warning if best epoch is different
        if val_logger is not None and val_logger.best_epoch != epoch_trained:
            logger.warning(
                f"The specified epoch was {epoch_trained} but the best epoch based on validation "
                f"set was {val_logger.best_epoch}. Ignore this warning if it was intentional."
            )

        # Load optimizer and scheduler state dicts if provided
        optimizer, scheduler = load_optimizer_and_scheduler(checkpoint, config.device, optimizer, scheduler)

        logger.info("Done.")

    else:
        raise FileNotFoundError(f"No {checkpoint_type} checkpoint found at '{checkpoint_path}'.")

    # Prepare dict to be returned
    return_dict = {
        "model": model,
        "config": config,
        "epoch": epoch_trained,
        "train_logger": train_logger,
        "val_logger": val_logger,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    return return_dict


def load_optimizer_and_scheduler(
    checkpoint: _StringDict,
    device: _Device,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[object] = None,
) -> Tuple[Optional[Optimizer], Optional[object]]:
    """
    Load the state dict of a given optimizer
    and scheduler (if they're provided).
    Helper function for `load_model()`.
    """

    def load_state_dict(
        obj: Union[Optimizer, object], key: str = "optimizer"
    ) -> Union[Optional[Optimizer], Optional[object]]:
        """
        Properly load state dict of optimizer/scheduler.
        """
        state_dict = checkpoint.get(key)
        if state_dict is not None:
            obj.load_state_dict(state_dict)
        else:
            raise KeyError(f"{key} argument expected its state dict in the loaded checkpoint but none was found.")
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
    config: _Config,
    epoch: Optional[int],
    config_info_dict: Optional[_StringDict] = None,
    checkpoint_type: Optional[str] = "state",
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

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model_name, epoch, config_info_dict)
    checkpoint_path = get_file_path(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logger.info(f"Removing {checkpoint_type} checkpoint '{checkpoint_path}'...")
        remove_object(checkpoint_path)
        logger.info("Done.")


def get_checkpoint_type_from_file(checkpoint_file: str) -> str:
    """
    Extract the checkpoint type from the file name provided.
    """
    checkpoint_type = None
    try:
        checkpoint_type = re.search(r"checkpoint-(?P<checkpoint_type>\b\w+\b)-", checkpoint_file).group(
            "checkpoint_type"
        )
    finally:
        validate_checkpoint_type(checkpoint_type)
    return checkpoint_type


def validate_checkpoint_type(checkpoint_type: str) -> None:
    """
    Check that the passed `checkpoint_type` is valid.
    """
    ALLOWED_CHECKPOINT_TYPES = ["state", "model"]
    assert checkpoint_type in ALLOWED_CHECKPOINT_TYPES, (
        f"'checkpoint_type' ('{checkpoint_type}') not understood (likely "
        f"from the file name provided). It must be one of {ALLOWED_CHECKPOINT_TYPES}."
    )


class EarlyStopping:
    """
    Implements early stopping in PyTorch.
    Reference: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
               with a few improvements.
    Common metrics (mse, accuracy, etc.) are
    ineherently supported, so specifying
    their params is optional.
    """

    SUPPORTED_CRITERIA = EvalCriteria()
    SUPPORTED_MODES = {"minimize": 0.0, "maximize": 1.0}
    CRITERIA_MODE_DICT = {metric: "maximize" for metric in SUPPORTED_CRITERIA.names}  # Default mode is maximize
    CRITERIA_MODE_DICT["mse"] = "minimize"  # Override to minimize for mse
    DEFAULT_PARAMS = {"min_delta": 0.2 * 1e-3, "patience": 5, "best_val_tol": 5e-3}

    def __init__(
        self,
        criterion: Optional[str] = "accuracy",
        mode: Optional[str] = None,
        min_delta: Optional[float] = None,
        patience: Optional[int] = None,
        best_val: Optional[float] = None,
        best_val_tol: Optional[float] = None,
    ):
        """
        :param criterion: Name of early stopping criterion
                          If criterion is supported ineherently, you may choose to
                          not provide any of the other params and use the default ones.
        :param mode: Whether to "maximize" or "minimize" the `criterion`
        :param min_delta: Minimum difference in metric required to prevent early stopping
        :param patience: No. of epochs (or steps) over which to monitor early stopping
        :param best_val: Best possible value of metric (if any)
        :param best_val_tol: Tolerance when comparing metric to best_val
                             This must be provided if `best_val` is provided
        """
        self.criterion = self.SUPPORTED_CRITERIA.canonicalize(criterion)
        self._init_params(
            mode=mode,
            min_delta=min_delta,
            patience=patience,
            best_val=best_val,
            best_val_tol=best_val_tol,
        )
        self._validate_params()
        self.best: Optional[float] = None
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
        if self.criterion and (self.criterion in self.SUPPORTED_CRITERIA):
            mode = self.CRITERIA_MODE_DICT[self.criterion]
            best_val = self.SUPPORTED_MODES[mode]
            params = {**self.DEFAULT_PARAMS, "mode": mode, "best_val": best_val}

            # Keep specified params and fall back to default for others
            kwargs = {k: v if v is not None else params[k] for k, v in kwargs.items()}

        else:
            # Non-default params must be provided for unsupported criteria
            for k, v in kwargs.items():
                if k not in self.DEFAULT_PARAMS.keys():
                    assert v is not None, (
                        f"The only criteria currently supported by "
                        f"default are {self.SUPPORTED_CRITERIA}, "
                        f"and hence param '{k}' is required."
                    )

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
        if self.best is None:  # After first step
            return True
        if self.mode == "minimize":
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def stop(self, metric: float) -> bool:
        """
        Check if early stopping criterion met.
        """
        if self.best is None:  # First step
            self.best = metric
            return False

        if np.isnan(metric):
            return True

        if self.is_better(metric):  # Reset patience counter if better
            self.num_bad_epochs = 0
            self.best = metric
        else:  # Otherwise increment counter by 1
            self.num_bad_epochs += 1

        # Check if already reached max value
        if self.best_val is not None and np.abs(self.best_val - metric) < self.best_val_tol:
            return True

        # Check if patience counter reached
        if self.num_bad_epochs >= self.patience:
            return True

        return False
