import numpy as np
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from .utils import convert_tensor_to_numpy
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc
)
from .types import Tuple, Optional, _config, _loss_or_losses, _eval_criterion_or_criteria


REGRESSION_LOSS_CRITERIA = ["mse"]
CLASSIFICATION_LOSS_CRITERIA = ["cross-entropy", "focal-loss"]
LOSS_CRITERIA = REGRESSION_LOSS_CRITERIA + CLASSIFICATION_LOSS_CRITERIA

REGRESSION_EVAL_CRITERIA = ["mse"]
CLASSIFICATION_EVAL_CRITERIA = ["accuracy", "precision", "recall", "f1", "auc"]
EVAL_CRITERIA = REGRESSION_EVAL_CRITERIA + CLASSIFICATION_EVAL_CRITERIA


def get_loss_eval_criteria(
    config: _config,
    reduction: Optional[str] = "mean",
    reduction_val: Optional[str] = None
) -> Tuple[_loss_or_losses, _loss_or_losses, _eval_criterion_or_criteria]:
    """
    Define train and val loss and evaluation criteria.
    :param reduction_val: If None, a common `reduction` will be used
                          for both train and val losses, otherwise
                          the specified one for val loss.
    """
    # Add/update train loss reduction and get criterion
    train_loss_kwargs = {**config.loss_kwargs, "reduction": reduction}
    loss_criterion_train = get_loss_criterion(config, criterion=config.loss_criterion,
                                              **train_loss_kwargs)

    # Add/update val loss reduction and get criterion
    if reduction_val is None:
        reduction_val = reduction
    val_loss_kwargs = {**config.loss_kwargs, "reduction": reduction_val}
    loss_criterion_val = get_loss_criterion(config, criterion=config.loss_criterion,
                                             **val_loss_kwargs)

    eval_criteria = get_eval_criteria(config, config.eval_criteria,
                                      **config.eval_criteria_kwargs)
    return loss_criterion_train, loss_criterion_val, eval_criteria

def get_loss_criterion(
    config: _config,
    criterion: Optional[str] = "cross-entropy",
    **kwargs
) -> _loss_or_losses:
    """
    Get loss criterion function.
    """
    loss_criterion = get_loss_criterion_function(config, criterion=criterion, **kwargs)
    return loss_criterion

def get_eval_criteria(config: _config, criteria: str, **kwargs) -> _eval_criterion_or_criteria:
    """
    Get a dictionary of eval criterion functions.
    """
    is_multilabel = config.model_type == "classification" and \
                    config.classification_type == "multilabel"
    if is_multilabel:
        if not kwargs.get("multilabel_reduction"):
            raise ValueError("Param 'multilabel_reduction' must be provided.")
        multilabel_reduction = kwargs["multilabel_reduction"]

    eval_criteria_dict = OrderedDict()
    for criterion in criteria:
        criterion_kwargs = kwargs.get(criterion, {})
        if is_multilabel:
            criterion_kwargs = {**criterion_kwargs, "multilabel_reduction": multilabel_reduction}
        eval_fn = get_eval_criterion_function(config, criterion=criterion, **criterion_kwargs)
        eval_criteria_dict[criterion] = eval_fn
    return eval_criteria_dict

def get_loss_criterion_function(
    config: _config,
    criterion: Optional[str] = "cross-entropy",
    **kwargs
) -> _loss_or_losses:
    """
    Get the function for a given loss `criterion`.
    :param kwargs: Misc kwargs for the loss. E.g. -
                   - `dim` for CrossEntropyLoss
                   - `alpha` and `gamma` for FocalLoss.
                   If it's a multilabel setting,
                   `multilabel_reduction` must be provided:
                    Type of multilabel_reduction to be
                    performed on the list of losses for
                    each class. (default="sum").
                    Choices: "sum" | "mean"
    """
    # Check for multilabel classification
    if config.model_type == "classification":
        # TODO: Remove this after extending FocalLoss
        if criterion == "focal-loss" and config.classification_type != "binary":
            raise ValueError("FocalLoss is currently only supported for binary classification.")

        elif config.classification_type == "multilabel":
            if not kwargs.get("multilabel_reduction"):
                raise ValueError("Param 'multilabel_reduction' must be provided.")

            multilabel_reduction = kwargs.pop("multilabel_reduction")
            if multilabel_reduction == "sum":
                agg_func = torch.sum
            elif multilabel_reduction == "mean":
                agg_func = torch.mean
            else:
                raise ValueError(
                    f"Param 'multilabel_reduction' ('{multilabel_reduction}') "
                    f"must be one of ['sum', 'mean']."
                )

    # Get per-label loss
    if criterion == "mse":
        loss_criterion = nn.MSELoss(**kwargs)
    elif criterion == "cross-entropy":
        loss_criterion = nn.CrossEntropyLoss(**kwargs)
    elif criterion == "focal-loss":
        # Remove `reduction` from kwargs since it's not required for FocalLoss
        loss_criterion = FocalLoss(**{k: v for k, v in kwargs.items() if k != "reduction"})
    else:
        raise ValueError(f"Param 'criterion' ('{criterion}') must be one of {LOSS_CRITERIA}.")

    # Regression
    if config.model_type == "regression":
        return loss_criterion

    # Binary / Multiclass classification
    elif config.classification_type in ["binary", "multiclass"]:
        return loss_criterion

    # Multilabel classification
    else:
        return lambda output_hist, y_hist: \
               agg_func(torch.stack([loss_criterion(output_hist, y_hist[...,i]) \
                                     for i in range(y_hist.shape[-1])], dim=0))

def get_eval_criterion_function(
    config: _config,
    criterion: Optional[str] = "accuracy",
    **kwargs
) -> _eval_criterion_or_criteria:
    """
    Get the function for a given evaluation `criterion`.
    :param kwargs: Misc kwargs for the eval criterion.
                   Mostly used in multiclass settings. E.g. -
                   - `average` for f1, precision, recall
                   - `pos_label` for auc
                   If it's a multilabel setting,
                   `multilabel_reduction` must be provided:
                    Type of multilabel_reduction to be
                    performed on the list of metric values
                    for each class. (default="sum").
                    Choices: "sum" | "mean"
    """
    # Check for multilabel classification
    if config.model_type == "classification" and config.classification_type == "multilabel":
        multilabel_reduction = kwargs.pop("multilabel_reduction")
        if multilabel_reduction == "none":
            agg_func = np.array
        elif multilabel_reduction == "mean":
            agg_func = np.mean
        else:
            raise ValueError(
                f"Param 'multilabel_reduction' ('{multilabel_reduction}') "
                f"must be one of ['mean', 'none']."
            )

    # Get per-label eval criterion
    if criterion == "mse":
        eval_criterion = partial(get_mse_loss, **kwargs)
    elif criterion in CLASSIFICATION_EVAL_CRITERIA:
        eval_criterion = partial(get_class_eval_metric, criterion=criterion, **kwargs)
    else:
        raise ValueError(f"Param 'criterion' ('{criterion}') must be one of {EVAL_CRITERIA}.")

    # Regression
    if config.model_type == "regression":
        return eval_criterion

    # Binary / Multiclass classification
    elif config.classification_type in ["binary", "multiclass"]:
        return eval_criterion

    # Multilabel classification
    else:
        return lambda output_hist, y_hist: \
            agg_func([eval_criterion(output_hist, y_hist[...,i]) \
                      for i in range(y_hist.shape[-1])])

def get_mse_loss(output_hist: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Compute MSE loss.
    """
    assert y_true.shape == output_hist.shape
    mse = nn.MSELoss(**kwargs)(output_hist, y_true).item()
    return mse

def get_class_eval_metric(
    output_hist: torch.Tensor,
    y_true: torch.Tensor,
    criterion: Optional[str] = "accuracy",
    **kwargs
) -> float:
    """
    Get eval criterion for a single class.

    As required, get:
      - class with max probability (for discrete metrics like accuracy etc.)
      - probs for y=1 (for computing AUC)
    and return the metric value for the given class
    """
    y_predicted = output_hist[:,1] if criterion == "auc" else output_hist.max(dim=-1)[1]
    y_true, y_predicted = convert_tensor_to_numpy((y_true, y_predicted))
    assert y_true.shape == y_predicted.shape
    y_true = y_true.astype(int)

    if criterion == "auc":
        fpr, tpr, threshold = roc_curve(y_true, y_predicted.astype(float), **kwargs)
        return auc(fpr, tpr)

    # criterion is one of ["accuracy", "precision", "recall", "f1"]
    criterion_fn_dict = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score
    }
    criterion_fn = partial(criterion_fn_dict[criterion], **kwargs)
    return criterion_fn(y_true, y_predicted.astype(int))


class FocalLoss(nn.Module):
    """
    Implements the focal loss for binary classification (ignores regression).
    Paper: https://arxiv.org/pdf/1708.02002.pdf
    Code insipration: https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
    """
    # TODO: Extend this to multiclass
    def __init__(self, alpha: Optional[float] = 0.25, gamma: Optional[float] = 2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss between raw logits and binary targets.
        :param outputs: (tensor) binary class probabilities of size (batch_size, 2)
        :param y: (tensor) encoded target labels of size (batch_size)

        :return (tensor) loss = FocalLoss(outputs, y)
        """
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
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"
