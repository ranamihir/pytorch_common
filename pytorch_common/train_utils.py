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
                   save_object, remove_object, ModelTracker, get_checkpoint_name


def train_model(model, config, train_loader, val_loader, optimizer, loss_criterion_train, \
                loss_criterion_test, eval_criteria, train_logger, val_logger, device, \
                epochs, scheduler=None, early_stopping=None, config_info_dict=None, start_epoch=0):
    '''
    Perform the entire training routine.
      - Params `epochs` and `device` are deliberately not derived directly from config
        so as to be able to change it on the fly without modifying config.
      - `start_epoch` may be provided if a trained checkpoint is loaded into the model
        and training is to be resumed from that point.
    '''
    best_epoch, stop_epoch = 0, start_epoch
    best_checkpoint_file, best_model = '', None
    for epoch in range(1+start_epoch, epochs+1+start_epoch):
        try:
            # Train epoch
            train_losses = train_epoch(
                model=model,
                loss_criterion=loss_criterion_train,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                scheduler=scheduler if config.use_scheduler_after_step else None
            )

            # Test on training set
            _, eval_metrics_train = test_epoch(
                model=model,
                dataloader=train_loader,
                loss_criterion=loss_criterion_test,
                eval_criteria=eval_criteria,
                device=device,
                return_outputs=False
            )
            # Add train losses+eval metrics, and log them
            train_logger.add_and_log_metrics(train_losses, eval_metrics_train)

            # Test on val set
            val_loss, eval_metrics_val = test_epoch(
                model=model,
                dataloader=val_loader,
                loss_criterion=loss_criterion_test,
                eval_criteria=eval_criteria,
                device=device,
                return_outputs=False
            )
            # Add val loss+eval metrics, and log them
            val_logger.add_and_log_metrics(val_loss, eval_metrics_val)

            # Take scheduler step
            if config.use_scheduler_after_epoch:
                scheduler_step(scheduler, val_loss)

            # Perform early stopping and replace checkpoint if required
            if config.use_early_stopping:
                if early_stopping.is_better(val_logger.get_early_stopping_metric()):
                    logging.info('Saving current best model checkpoint and removing previous one...')
                    best_checkpoint_file = save_model(model, optimizer, config, \
                                                      train_logger, val_logger, epoch, \
                                                      config_info_dict, scheduler)
                    remove_model(config, best_epoch, config_info_dict)
                    logging.info('Done.')
                    best_epoch = epoch

                if early_stopping.stop(val_logger.get_early_stopping_metric()):
                    stop_epoch = epoch
                    logging.info(f'Stopping early after {stop_epoch} epochs.')
                    break

            else: # Save all checkpoints if early stopping not used
                save_model(model, optimizer, config, \
                           val_logger, train_logger, \
                           epoch, config_info_dict, scheduler)

            stop_epoch = epoch
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupted!')
            stop_epoch = epoch - 1
            break

    # Save the model checkpoints
    logging.info('Dumping model and results...')
    save_model(model, optimizer, config, train_logger, \
               val_logger, stop_epoch, config_info_dict, scheduler)

    # Save current and best models
    save_model(model.copy(), optimizer, config, train_logger, val_logger, \
               stop_epoch, config_info_dict, scheduler, checkpoint_type='model')
    if best_checkpoint_file != '':
        checkpoint = load_model(model.copy(), config, best_checkpoint_file, optimizer, scheduler)
        best_model = checkpoint['model']
        optimizer, scheduler = checkpoint['optimizer'], checkpoint['scheduler']
        checkpoint = None # Free up memory
        best_config_info_dict = {**config_info_dict, 'best': True}
        save_model(best_model, optimizer, config, train_logger, val_logger, \
                   stop_epoch, best_config_info_dict, scheduler, checkpoint_type='model')
    logging.info('Done.')

    return_dict = {
        'model': model,
        'best_model': best_model if best_model is not None else model,
        'train_logger': train_logger,
        'val_logger': val_logger,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'stop_epoch': stop_epoch,
        'best_epoch': best_epoch,
        'best_checkpoint_file': best_checkpoint_file
    }
    return return_dict

@timing
def train_epoch(model, dataloader, loss_criterion, optimizer, device, epoch, scheduler=None):
    '''
    Perform one training epoch and return the loss per example for each iteration
    :param scheduler: Pass this only if it's a scheduler that requires taking a step
                      after each batch iteration (e.g. CyclicLR), otherwise None
    Tip: During development, you can just override num_batches with 1 (or a small
         number) to run quickly on a small dataset.
    '''
    model.train()

    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Print 50 times in an epoch (or every time, if num_batches < 50)
    batches_to_print = np.unique(np.linspace(0, num_batches, num=50, endpoint=True, dtype=int))

    loss_hist = [] # Store all losses
    for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
        # Assume first two elements of batch are (inputs, target).
        # Just to be sure in case other things are
        # being passed in the batch for debugging.
        x, y = send_batch_to_device(batch[:2], device)

        optimizer.zero_grad()
        model.zero_grad()

        # Get model outputs
        outputs = model(x)
        outputs = get_model_outputs_only(outputs)

        # Backprop + clip gradients + take scheduler step
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

        # Print progess
        if batch_idx in batches_to_print:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * batch_size, num_examples,
                100. * (batch_idx+1) / num_batches, loss_value))

    optimizer.zero_grad()
    model.zero_grad()
    return loss_hist

@timing
@torch.no_grad()
def test_epoch(model, dataloader, loss_criterion, eval_criteria, device, return_outputs=False):
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
    Make predictions on entire dataset and return raw outputs and optionally
    class predictions and probabilities if it's a classification model
    '''
    model.eval()

    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Print 50 times in an epoch (or every time, if num_batches < 50)
    batches_to_print = np.unique(np.linspace(0, num_batches, num=50, endpoint=True, dtype=int))

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

        # Print progess
        if batch_idx in batches_to_print:
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
    model.eval()

    num_batches, num_examples = len(dataloader), len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Print 50 times in an epoch (or every time, if num_batches < 50)
    batches_to_print = np.unique(np.linspace(0, num_batches, num=50, endpoint=True, dtype=int))

    all_embeddings = []
    seq_pooler = SequencePooler(config.model)

    for batch_idx, batch in enumerate(islice(dataloader, num_batches)):
        x = send_batch_to_device(batch[0], config.device) # Only need inputs for getting embeddings
        batch_embeddings = seq_pooler(model(x))
        batch_embeddings = convert_tensor_to_numpy(batch_embeddings).tolist()
        all_embeddings.extend(batch_embeddings)

        # Print progess
        if batch_idx in batches_to_print:
            logging.info('{}/{} ({:.0f}%) complete.'.format(
                (batch_idx+1) * batch_size, num_examples,
                100. * (batch_idx+1) / num_batches))

    return np.array(all_embeddings)

def scheduler_step(scheduler, val_metric=None):
    '''
    Take a scheduler step.
    Some schedulers, e.g. `ReduceLROnPlateau` require
    the validation metric to take a step, while (most)
    others don't.
    '''
    scheduler_name = scheduler.__class__.__name__
    REQUIRE_VAL_METRIC = ['ReduceLROnPlateau']
    if scheduler_name in REQUIRE_VAL_METRIC:
        assert val_metric is not None, \
            f'Param "val_metric" must be provided for {scheduler_name} scheduler.'
        scheduler.step(val_metric)
    else:
        scheduler.step()

def save_model(model, optimizer, config, train_logger, val_logger, \
               epoch, misc_info=None, scheduler=None, checkpoint_type='state'):
    '''
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
                            Choices = 'state' | model'
                            Default = 'state'
    :returns name of checkpoint file
    '''
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type)

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model, epoch, misc_info)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
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

    # `dill` is used when a model has serialization issues, e.g. that
    # caused by having a lambda function as an attribute of the model.
    # Regular pickling won't work, but it will with dill.
    # Note 1: Avoid using it if possible since it's a little slower.
    # Note 2: It has more robust serialization though.
    # Note 3: When `checkpoint_type='state'`, it should automatically
    #         always work with pickle.
    try:
        torch.save(checkpoint, checkpoint_path)
    except AttributeError:
        torch.save(checkpoint, checkpoint_path, pickle_module=dill)

    logging.info('Done.')
    return checkpoint_file

def generate_checkpoint_dict(optimizer, config, train_logger, \
                             val_logger, epoch, scheduler=None):
    '''
    Generate a dictionary for storing a checkpoint.
    Helper function for `save_model()`.
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

def load_model(model, config, checkpoint_file, optimizer=None, \
               scheduler=None, checkpoint_type='state'):
    '''
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
                            Choices = 'state' | model'
                            Default = 'state'
    '''
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type, checkpoint_file)

    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logging.info(f'Loading {checkpoint_type} checkpoint "{checkpoint_path}"...')

        # See `save_model()` for explanation
        try:
            checkpoint = torch.load(checkpoint_path)
        except AttributeError:
            checkpoint = torch.load(checkpoint_path, pickle_module=dill)

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

        # Load config
        config = checkpoint['config']

        # Extract last trained epoch from checkpoint file
        epoch_trained = int(os.path.splitext(checkpoint_file)[0].split('-epoch_')[-1])

        # Load train / val loggers
        train_logger = checkpoint['train_logger']
        val_logger = checkpoint['val_logger']

        # Verify consistency of last epoch trained
        assert epoch_trained == checkpoint['epoch'] == max(train_logger.epochs)

        # Load optimizer and scheduler state dicts
        optimizer, scheduler = load_optimizer_and_scheduler(checkpoint, config.device, \
                                                            optimizer, scheduler)

        logging.info('Done.')

    else:
        raise FileNotFoundError(f'No {checkpoint_type} checkpoint found at "{checkpoint_path}".')

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
    Helper function for `load_model()`.
    '''
    # Load optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer = send_optimizer_to_device(optimizer, device)

    # Load scheduler
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler

def remove_model(config, epoch, misc_info=None, checkpoint_type='state'):
    '''
    Remove a checkpoint/model at a given epoch.
    Used in early stopping if better performance
    is observed at a subsequent epoch.

    :param checkpoint_type: Type of checkpoint to load
                            Choices = 'state' | model'
                            Default = 'state'
    '''
    # Validate checkpoint_type
    validate_checkpoint_type(checkpoint_type)

    checkpoint_file = get_checkpoint_name(checkpoint_type, config.model, epoch, misc_info)
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_file)
    if os.path.isfile(checkpoint_path):
        logging.info(f'Removing {checkpoint_type} checkpoint "{checkpoint_path}"...')
        remove_object(checkpoint_path)
        logging.info('Done.')

def validate_checkpoint_type(checkpoint_type, checkpoint_file=None):
    '''
    Check that the passed `checkpoint_type` is valid and matches that
    obtained from `checkpoint_file`, if provided.
    '''
    allowed_checkpoint_types = ['state', 'model']
    assert checkpoint_type in allowed_checkpoint_types, f'Param "checkpoint_type" ("{checkpoint_type}")'\
                                                        f' must be one of {allowed_checkpoint_types}.'

    # Check that provided checkpoint_type matches that of checkpoint_file
    if checkpoint_file is not None:
        file_checkpoint_type = checkpoint_file.split('-', 3)[1]
        assert file_checkpoint_type == checkpoint_type, \
            f'The type of checkpoint provided in param "checkpoint_type" ("{checkpoint_type}") does '\
            f'not match that obtained from the model at "{checkpoint_file}" ("{file_checkpoint_type}").'


class EarlyStopping(object):
    '''
    Implements early stopping in PyTorch.
    Reference: https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
               with a few improvements.
    '''
    SUPPORTED_CRITERIA = ['mse', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    SUPPORTED_MODES = {'minimize': 0., 'maximize': 1.}
    CRITERIA_MODE_DICT = {
        'mse': 'minimize',
        'accuracy': 'maximize',
        'precision': 'maximize',
        'recall': 'maximize',
        'f1': 'maximize',
        'auc': 'maximize'
    }
    DEFAULT_PARAMS = {'min_delta': 0.2*1e-3, 'patience': 5, 'best_val_tol': 5e-3}

    def __init__(self, criterion='f1', mode=None, min_delta=None, \
                 patience=None, best_val=None, best_val_tol=None):
        '''
        :param criterion: name of early stopping criterion
                          If criterion is supported ineherently, you may choose to
                          not provide any of the other params and use the default ones.
        :param mode: whether to 'maximize' or 'minimize' the `criterion`
        :param min_delta: Minimum difference in metric required to prevent early stopping
        :param patience: No. of epochs (or steps) over which to monitor early stopping
        :param best_val: Best possible value of metric (if any)
        :param best_val_tol: Tolerance when comparing metric to best_val
        '''
        self.criterion = criterion
        self._init_params(mode=mode, min_delta=min_delta, patience=patience, \
                          best_val=best_val, best_val_tol=best_val_tol)
        self._validate_params()
        self.best = None
        self.num_bad_epochs = 0

        if self.patience == 0:
            self.is_better = lambda metric: True
            self.stop = lambda metric: False

    def _init_params(self, **kwargs):
        if self.criterion in self.SUPPORTED_CRITERIA:
            mode = self.CRITERIA_MODE_DICT[self.criterion]
            best_val = self.SUPPORTED_MODES[mode]
            params = {**self.DEFAULT_PARAMS, 'mode': mode, 'best_val': best_val}
            kwargs = {k: v if v is not None else params[k] for k, v in kwargs.items()}
        else:
            for k, v in kwargs.items():
                if k not in self.DEFAULT_PARAMS.keys():
                    assert v is not None, f'The only criteria currently supported by '\
                                          f'default are {self.SUPPORTED_CRITERIA}, '\
                                          f'and hence param "{k}" is required.'

        # Finally set values and validate them
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _validate_params(self):
        '''
        Check validity of mode of optimization
        '''
        if self.mode not in self.SUPPORTED_MODES.keys():
            raise ValueError(f'Mode "{self.mode}" is unknown.')
        if self.best_val is not None and self.best_val_tol is None:
            raise ValueError('Param "best_val_tol" must be provided if "best_val" is provided.')

    def is_better(self, metric):
        '''
        Check if the provided `metric` is the best one so far
        '''
        if self.best is None:
            return True
        if self.mode == 'minimize':
            return metric < self.best - self.min_delta
        return metric > self.best + self.min_delta

    def stop(self, metric):
        '''
        Check if early stopping criterion met
        '''
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
        if self.best_val is not None and np.abs(self.best_val-metric) < self.best_val_tol:
            return True

        if self.num_bad_epochs >= self.patience:
            return True

        return False
