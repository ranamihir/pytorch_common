import numpy as np
import pandas as pd
import logging
from vrdscommon import timing
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from .utils import convert_tensor_to_numpy
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc


def get_loss_criterion(config, criterion='cross-entropy', **kwargs):
    '''
    Get loss criterion function
    '''
    loss_criterion = set_loss_criterion_function(config, criterion=criterion, **kwargs)
    return loss_criterion

def get_eval_criteria(config, criteria, **kwargs):
    '''
    Get a dictionary of eval criterion functions
    '''
    is_multilabel = config.model_type == 'classification' and config.classification_type == 'multilabel'
    if is_multilabel:
        if not kwargs.get('multilabel_reduction'):
            raise ValueError('Param "multilabel_reduction" must be provided.')
        multilabel_reduction = kwargs['multilabel_reduction']

    eval_criteria_dict = OrderedDict()
    for criterion in criteria:
        criterion_kwargs = kwargs.get(criterion, {})
        if is_multilabel:
            criterion_kwargs = {**criterion_kwargs, 'multilabel_reduction': multilabel_reduction}
        eval_fn = set_eval_criterion_function(config, criterion=criterion, **criterion_kwargs)
        eval_criteria_dict[criterion] = eval_fn
    return eval_criteria_dict

def set_loss_criterion_function(config, criterion='cross-entropy', **kwargs):
    '''
    :param kwargs: Misc kwargs for the loss. E.g. -
                   - `dim` for CrossEntropyLoss
                   - `alpha` and `gamma` for FocalLoss.
                   If it's a multilabel setting,
                   `multilabel_reduction` must be provided:
                    Type of multilabel_reduction to be
                    performed on the list of losses for
                    each class. (default='sum').
                    Choices: 'sum' | 'mean'
    '''
    # Check for multilabel classification
    if config.model_type == 'classification' and config.classification_type == 'multilabel':
        if not kwargs.get('multilabel_reduction'):
            raise ValueError('Param "multilabel_reduction" must be provided.')

        multilabel_reduction = kwargs.pop('multilabel_reduction')
        if multilabel_reduction == 'sum':
            agg_func = torch.sum
        elif multilabel_reduction == 'mean':
            agg_func = torch.mean
        else:
            raise ValueError(f'Param "multilabel_reduction" ("{multilabel_reduction}") '\
                             f'must be one of ["sum", "mean"].')

    # Get per-label loss
    allowed_losses = ['mse', 'cross-entropy', 'focal-loss']
    if criterion == 'mse':
        loss_criterion = nn.MSELoss(**kwargs)
    elif criterion == 'cross-entropy':
        loss_criterion = nn.CrossEntropyLoss(**kwargs)
    elif criterion == 'focal-loss':
        loss_criterion = FocalLoss(**kwargs)
    else:
        raise ValueError(f'Param "criterion" ("{criterion}") must be one of {allowed_losses}.')

    # Regression
    if config.model_type == 'regression':
        return loss_criterion

    # Binary / Multiclass classification
    elif config.classification_type in ['binary', 'multiclass']:
        return loss_criterion

    # Multilabel classification
    else:
        return lambda output_hist, y_hist: \
               agg_func(torch.stack([loss_criterion(output_hist[...,i], y_hist[...,i]) \
                                     for i in range(y_hist.shape[-1])], dim=0))

def set_eval_criterion_function(config, criterion='accuracy', **kwargs):
    '''
    :param kwargs: Misc kwargs for the eval criterion.
                   Mostly used in multiclass settings. E.g. -
                   - `average` for f1, precision, recall
                   - `pos_label` for auc
                   If it's a multilabel setting,
                   `multilabel_reduction` must be provided:
                    Type of multilabel_reduction to be
                    performed on the list of metric values
                    for each class. (default='sum').
                    Choices: 'sum' | 'mean'
    '''
    # Check for multilabel classification
    if config.model_type == 'classification' and config.classification_type == 'multilabel':
        multilabel_reduction = kwargs.pop('multilabel_reduction')
        if multilabel_reduction == 'none':
            agg_func = np.array
        elif multilabel_reduction == 'mean':
            agg_func = np.mean
        else:
            raise ValueError(f'Param "multilabel_reduction" ("{multilabel_reduction}") '\
                             f'must be one of ["mean", "none"].')

    # Get per-label eval criterion
    allowed_criteria = ['mse', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    if criterion == 'mse':
        eval_criterion = get_mse_loss
    elif criterion in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        eval_criterion = partial(get_class_eval_metric, criterion=criterion, **kwargs)
    else:
        raise ValueError(f'Param "criterion" ("{criterion}") must be one of {allowed_criteria}.')

    # Regression
    if config.model_type == 'regression':
        return eval_criterion

    # Binary / Multiclass classification
    elif config.classification_type in ['binary', 'multiclass']:
        return eval_criterion

    # Multilabel classification
    else:
        return lambda output_hist, y_hist: agg_func([eval_criterion(output_hist[:,i], y_hist[:,i]) \
                                                     for i in range(y_hist.shape[1])])

@torch.no_grad()
def get_mse_loss(output_hist, y_true):
    '''
    Get MSE loss
    '''
    assert y_true.shape == y_predicted.shape
    mse = nn.MSELoss()(y_predicted, y_true).item()
    return mse

@torch.no_grad()
def get_class_eval_metric(output_hist, y_true, criterion='accuracy', **kwargs):
    '''
    Get eval criterion for a single class

    As required, get:
      - class with max probability (for discrete metrics like accuracy etc.)
      - probs for y=1 (for computing AUC)
    and return the metric value for the given class
    '''
    y_predicted = output_hist[:,1] if criterion == 'auc' else output_hist.max(dim=-1)[1]
    y_true, y_predicted = convert_label_tensors_to_numpy(y_true, y_predicted)
    y_true = y_true.astype(int)

    if criterion == 'auc':
        fpr, tpr, threshold = roc_curve(y_true, y_predicted.astype(float), **kwargs)
        return auc(fpr, tpr)

    # criterion is one of ['accuracy', 'precision', 'recall', 'f1']
    criterion_fn_dict = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }
    criterion_fn = partial(criterion_fn_dict[criterion], **kwargs)
    return criterion_fn(y_true, y_predicted.astype(int))

def convert_label_tensors_to_numpy(y_true, y_predicted):
    '''
    Move both true labels and prediction arrays
    to numpy for downstream metric computation
    '''
    y_true, y_predicted = convert_tensor_to_numpy((y_true, y_predicted))
    assert y_true.shape == y_predicted.shape
    return y_true, y_predicted


class FocalLoss(nn.Module):
    '''
    Implements the focal loss for binary classification (ignores regression).
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Code insipration: https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
    '''
    # TODO: Extend this to multiclass
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, y):
        '''
        Compute the focal loss between raw logits and binary targets.
        :param outputs: (tensor) binary class probabilities of size (batch_size, 2)
        :param y: (tensor) encoded target labels of size (batch_size)

        :return (tensor) loss = FocalLoss(outputs, y)
        '''
        probs1 = torch.sigmoid(outputs[:,1])
        targets = y.float()

        # alpha balancing weights = alpha if y = 1 else (1-alpha)
        w_alpha = torch.ones_like(targets, device=targets.device) * self.alpha
        w_alpha = torch.where(torch.eq(targets, 1.), w_alpha, 1. - w_alpha)

        # focal weights = (1-p)^gamma if y = 1 else p^gamma
        w_focal = torch.where(torch.eq(targets, 1.), 1. - probs1, probs1)
        w_focal = torch.pow(w_focal, self.gamma)

        # Focal loss = w_alpha * w_focal * BCELoss
        bce_loss = - (targets * probs1.log() + (1. - targets) * (1. - probs1).log())
        focal_loss = w_alpha * w_focal * bce_loss

        return focal_loss.mean()

    def __repr__(self):
        return f'FocalLoss(alpha={self.alpha}, gamma={self.gamma})'
