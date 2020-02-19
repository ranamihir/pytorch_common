import numpy as np
import pandas as pd
import os
import logging
import dill
from itertools import islice
from vrdscommon import timing

import torch
import torch.nn as nn

from .utils import get_model_outputs_only, send_batch_to_device, send_model_to_device, \
                   send_optimizer_to_device, convert_tensor_to_numpy, SequencePooler, \
                   save_object, ModelTracker, get_checkpoint_name


@timing
def train(model, dataloader, loss_criterion, optimizer, device, epoch, scheduler=None):
    '''
    Perform one training epoch and return the loss per example for each iteration
    :param scheduler: Pass this only if it's a scheduler that requires taking a step
                      after each batch iteration (e.g. CyclicLR), otherwise None
    Tip: During development, you can just override num_batches with 1 (or a small
         number) to run quickly on a small dataset.
    '''
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = np.ceil(num_examples / num_batches).astype(int)
    model.train()

    loss_hist = [] # Store all losses
    for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
        # Assume first two elements of batch are (inputs, target).
        # Just to be sure in case other things are
        # being passed in the batch for debugging.
        x, y = send_batch_to_device(batch[:2], device)

        optimizer.zero_grad()

        outputs = model(x)
        outputs = get_model_outputs_only(outputs)

        loss = loss_criterion(outputs, y)
        loss_value = loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.)
        optimizer.step()
        if scheduler is not None:
            scheduler_step(scheduler, loss_value)

        # Accurately compute loss, because of different batch size
        loss_train = loss_value * batch_size / num_examples
        loss_hist.append(loss_train)

        # Print 50 times in a batch; if dataset too small print every time (to avoid division by 0)
        if (batch_idx+1) % max(1, (num_examples//(50*batch_size))) == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * batch_size, num_examples,
                100. * (batch_idx+1) / num_batches, loss_value))

    optimizer.zero_grad()
    return loss_hist

@timing
@torch.no_grad()
def test(model, dataloader, loss_criterion, eval_criteria, device, return_outputs=False):
    '''
    Perform evaluation on the entire dataset and return the
    loss per example and all eval criteria on whole dataset.
    Tip: During development, you can just override num_batches with 1 (or a small
         number) to run quickly on a small dataset.
    '''
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    model.eval()

    loss_hist, y_hist, outputs_hist = [], [], [] # Store all losses, target, and outputs
    for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
        # Assume first two elements of batch are (inputs, target).
        # Just to be sure in case other things are
        # being passed in the batch for debugging.
        x, y = send_batch_to_device(batch[:2], device)
        outputs = model(x)
        outputs = get_model_outputs_only(outputs)

        loss = loss_criterion(outputs, y)
        loss_test = loss.item() / num_examples
        loss_hist.append(loss_test)

        outputs_hist.append(outputs)
        y_hist.append(y)

    outputs_hist = send_batch_to_device(torch.cat(outputs_hist, dim=0), 'cpu')
    y_hist = send_batch_to_device(torch.cat(y_hist, dim=0), 'cpu')

    # Compute all evaluation criteria
    eval_metrics = {eval_criterion: eval_fn(outputs_hist, y_hist) \
                    for eval_criterion, eval_fn in eval_criteria.items()}

    if return_outputs:
        return np.sum(loss_hist), eval_metrics, outputs_hist, y_hist
    return np.sum(loss_hist), eval_metrics

@torch.no_grad()
def get_all_predictions(model, dataloader, device, threshold_prob=None):
    '''
    Make predictions on entire dataset using model's own
    `predict()` method and return raw outputs and optionally
    class predictions and probabilities if it's a classification task
    '''
    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = np.ceil(num_examples / num_batches).astype(int)
    model.eval()

    outputs_hist, preds_hist, probs_hist = [], [], []
    for batch_idx, batch in enumerate(islice(dataloader, len(dataloader))):
        x = send_batch_to_device(batch[0], device) # Only need inputs for making predictions

        outputs = get_model_outputs_only(model(x))
        outputs = send_batch_to_device(outputs, 'cpu')
        outputs_hist.extend(outputs)
        if model.model_type == 'classification': # Get class predictions and probabilities
            preds, probs = model.predict_proba(outputs, threshold_prob)
            preds_hist.extend(preds)
            probs_hist.extend(probs)

        # Print 50 times in a batch; if dataset too small print every time (to avoid division by 0)
        if (batch_idx+1) % max(1, (num_examples//(50*batch_size))) == 0:
            logging.info('{}/{} ({:.0f}%) complete.'.format(
                (batch_idx+1) * batch_size, num_examples,
                100. * (batch_idx+1) / num_batches))

    outputs_hist = torch.stack(outputs_hist, dim=0)
    if model.model_type == 'classification':
        preds_hist = torch.stack(preds_hist, dim=0)
        probs_hist = torch.stack(probs_hist, dim=0)
    return outputs_hist, preds_hist, probs_hist

@timing
@torch.no_grad()
def get_all_embeddings(model, dataloader, config):
    '''
    Get embeddings for all examples in the dataset
    '''
    num_batches = len(dataloader)
    model.eval()

    all_embeddings = []
    seq_pooler = SequencePooler(config.model)

    for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
        x = send_batch_to_device(batch[0], config.device) # Only need inputs for getting embeddings
        batch_embeddings = seq_pooler(model(x))
        batch_embeddings = convert_tensor_to_numpy(batch_embeddings).tolist()
        all_embeddings.extend(batch_embeddings)

        if (batch_idx+1) % 50 == 0:
            logging.info(f'Batch {batch_idx+1}/{num_batches} completed.')

    return np.array(all_embeddings)

def scheduler_step(scheduler, val_metric=None):
    '''
    Take a scheduler step.
    Some schedulers, e.g. `ReduceLROnPlateau` require
    the validation metric to take a step, while (most)
    others don't.
    '''
    require_val_metric = ['ReduceLROnPlateau']
    scheduler_name = scheduler.__class__.__name__
    if scheduler_name in require_val_metric:
        assert val_metric is not None, \
            f'Param "val_metric" must be provided for {scheduler_name} scheduler'
        scheduler.step(val_metric)
    else:
        scheduler.step()

def save_checkpoint(model, optimizer, config, train_logger, \
                    val_logger, epoch, misc_info=None, scheduler=None):
    '''
    Save the checkpoint at a given epoch.
    It saves the following variables:
        - Model state dict
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    '''
    return save_checkpoint_common('state', model, optimizer, config, train_logger, \
                                  val_logger, epoch, misc_info, scheduler)

def save_model(model, optimizer, config, train_logger, \
                val_logger, epoch, misc_info=None, scheduler=None):
    '''
    Similar to `save_checkpoint()`, except it
    saves the entire model (copied to CPU).
    Not recommended since it breaks if the
    model code is changed.
    It saves the following variables:
        - (Entire) model
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    '''
    return save_checkpoint_common('model', model, optimizer, config, train_logger, \
                                  val_logger, epoch, misc_info, scheduler)

def save_checkpoint_common(checkpoint_type, model, optimizer, config, train_logger, \
                           val_logger, epoch, misc_info=None, scheduler=None):
    '''
    Common function to save the checkpoint at a given epoch.
    It can save either the entire model or just its state dict.
    Additionally, it saves the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    '''
    checkpoint_name = get_checkpoint_name(checkpoint_type, config.model, epoch, misc_info)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    logging.info(f'Saving {checkpoint_type} checkpoint "{checkpoint_path}"...')

    # Generate appropriate checkpoint dictionary
    checkpoint = generate_checkpoint_dict(optimizer, config, train_logger, \
                                          val_logger, epoch, scheduler)

    # Save model in appropriate way
    if checkpoint_type == 'state':
        checkpoint['model'] = model.module.state_dict() if hasattr(model, 'module') \
                              else model.state_dict()
    else:
        checkpoint['model'] = send_model_to_device(model, 'cpu') # Save model on CPU

    '''
    `dill` is used when a model has serialization issues, e.g. that
    caused by having a lambda function as an attribute of the model.
    Regular pickling won't work, but it will with dill.
    Note 1: Avoid using it if possible since it's a little slower.
    Note 2: It has more robust serialization though.
    Note 3: When `checkpoint_type='state'`, it should automatically
            always work with pickle.
    '''
    try:
        torch.save(checkpoint, checkpoint_path)
    except AttributeError:
        torch.save(checkpoint, checkpoint_path, pickle_module=dill)

    logging.info('Done.')
    return checkpoint_path

def generate_checkpoint_dict(optimizer, config, train_logger, \
                             val_logger, epoch, scheduler=None):
    '''
    Generate a dictionary for storing a checkpoint.
    Helper function for `save_checkpoint_common()`.
    It saves the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    '''
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'config': config, # Good practice to store config too
        'train_logger': train_logger,
        'val_logger': val_logger,
        'epoch': epoch,
    }
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    return checkpoint

def remove_checkpoint(config, epoch, misc_info=None, checkpoint_type='state'):
    '''
    Remove a checkpoint at a given epoch.
    Used in early stopping if better performance
    is observed at a subsequent epoch.
    '''
    checkpoint_name = get_checkpoint_name(checkpoint_type, config.model, epoch, misc_info)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)
    if os.path.exists(checkpoint_path):
        logging.info(f'Removing {checkpoint_type} checkpoint "{checkpoint_path}"...')
        os.remove(checkpoint_path)
        logging.info('Done.')

def load_checkpoint(model, config, checkpoint_file, optimizer=None, scheduler=None):
    '''
    Load a checkpoint saved using `save_checkpoint()`.
    Additionally, it loads the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    Note: Input model, optimizer, and scheduler should
          be pre-defined. This routine only updates
          their states.
    '''
    return load_checkpoint_common('state', model, config, checkpoint_file, optimizer, scheduler)

def load_model(config, checkpoint_file, optimizer=None, scheduler=None):
    '''
    Load a model saved using `save_model()`.
    Additionally, it loads the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    Note: Input model, optimizer, and scheduler should
          be pre-defined. This routine only updates
          their states.
    '''
    return load_checkpoint_common('model', None, config, checkpoint_file, optimizer, scheduler)

def load_checkpoint_common(checkpoint_type, model, config, checkpoint_file, optimizer=None, scheduler=None):
    '''
    Common function to load the checkpoint at a given epoch.
    It can load either the entire model or just its state dict.
    Additionally, it loads the following variables:
        - Optimizer and scheduler (if provided) state dicts
        - Epoch number
        - History of train and validation
          losses and eval metrics so far
        - Current training config
    Note: Input model, optimizer, and scheduler should
          be pre-defined. This routine only updates
          their states.
    '''
    # Set default values if no checkpoint found
    train_logger, val_logger = None, None
    epoch_trained = 0

    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logging.info(f'Loading {checkpoint_type} checkpoint "{checkpoint_path}"...')

        # See `save_checkpoint_common()` for explanation
        try:
            checkpoint = torch.load(checkpoint_path)
        except AttributeError:
            checkpoint = torch.load(checkpoint_path, pickle_module=dill)

        # Load config
        config = checkpoint['config']

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(os.path.splitext(checkpoint_file)[0].split('-epoch_')[-1])
        assert epoch_trained == checkpoint['epoch']

        # Load optimizer and scheduler state dicts
        optimizer, scheduler = load_optimizer_and_scheduler(checkpoint, config.device, \
                                                            optimizer, scheduler)

        # Load train / val loggers
        train_logger = checkpoint['train_logger']
        val_logger = checkpoint['val_logger']
        assert epoch_trained == max(train_logger.epochs)

        # Load model in appropriate way
        if checkpoint_type == 'state': # Load state dict
            assert model is not None
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint['model'])
        else: # Load entire model
            assert model is None
            model = checkpoint['model']

        logging.info('Done.')

    else:
        raise FileNotFoundError(f'No {checkpoint_type} checkpoint found at "{checkpoint_path}"!')

    # Prepare dict to be returned
    return_dict = {
        'model': model,
        'optimizer': optimizer,
        'config': config,
        'train_logger': train_logger,
        'val_logger': val_logger,
        'epoch': epoch_trained,
        'scheduler': scheduler
    }

    return return_dict

def load_optimizer_and_scheduler(checkpoint, device, optimizer=None, scheduler=None):
    '''
    Load the state dict of a given optimizer
    and scheduler, if they're provided.
    Helper function for `load_checkpoint_common()`.
    '''
    # Load optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer = send_optimizer_to_device(optimizer, device)

    # Load scheduler
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler


class EarlyStopping(object):
    '''
    Implements early stopping in PyTorch.
    Reference: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
               with minor improvements.
    '''
    def __init__(self, mode='minimize', min_delta=0, patience=10, max_val=None, max_val_tol=None):
        '''
        :param min_delta: Minimum difference in metric required to prevent early stopping
        :param patience: No. of epochs (or steps) over which to monitor early stopping
        :param max_val: Maximum possible value of metric (if any)
        :param max_val_tol: Tolerance when comparing metric to max_val
        '''
        self.mode = mode
        self._check_mode()
        self.min_delta = min_delta
        self.patience = patience
        self.max_val = max_val
        self.max_val_tol = max_val_tol

        self.best = None
        self.num_bad_epochs = 0

        if patience == 0:
            self.is_better = lambda metric: True
            self.stop = lambda metric: False

        if self.max_val is not None:
            assert self.max_val_tol is not None, 'Max value tolerance must be provided.'

    def _check_mode(self):
        if self.mode not in ['maximize', 'minimize']:
            raise ValueError(f'mode "{self.mode}" is unknown!')

    def is_better(self, metric):
        if self.best is None:
            return True
        if self.mode == 'minimize':
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def stop(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if np.isnan(metric):
            return True

        if self.is_better(metric):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_bad_epochs += 1

        # Check if already reached max value
        if self.max_val and np.abs(self.max_val - metric) < self.max_val_tol:
            return True

        if self.num_bad_epochs >= self.patience:
            return True

        return False
