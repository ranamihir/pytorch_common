"""
To add a new loss criterion:
  1. Add it to `REGRESSION_LOSS_CRITERIA` or `CLASSIFICATION_LOSS_CRITERIA` at the bottom as required.
  2. Update the logic in `metrics.get_loss_criterion_function()`.
  3. Add the actual function for metric computation here.

To add a new evaluation criterion:
  1. Add it to `EVAL_METRIC_FUNCTIONS` at the bottom.
  2. Add the actual function for metric computation here.
"""
import re

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve

from .types import *

TOP_K_ACCURACY_REGEX = "top_([0-9]+)_accuracy"


def dummy_preprocessing(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> _TensorOrTensors:
    """
    Dummy preprocessing function that does
    nothing but follow the required API.
    """
    return y_predicted, y_true


def highest_prob_class(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> _TensorOrTensors:
    """
    Get the index of class with highest probability.
    """
    y_predicted = y_predicted.max(dim=-1)[1]
    assert y_true.shape == y_predicted.shape
    return y_predicted, y_true


def prob_class1(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> _TensorOrTensors:
    """
    Get the probabilities of class at index 1.
    """
    y_predicted = y_predicted[:, 1]
    assert y_true.shape == y_predicted.shape
    return y_predicted, y_true


def correct_argument_order(func: Callable) -> Callable:
    """
    PyTorch metrics normally take `(y_predicted, y_true)`
    as inputs, while sklearn metrics take `(y_true, y_predicted)`
    as input.
    This function reverses the order (used for sklearn metrics)
    so that all metrics have a consistent API.
    """
    corrected_func = lambda y_true, y_predicted, *args, **kwargs: func(y_predicted, y_true, *args, **kwargs)
    return corrected_func


def get_mse_loss(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Compute MSE loss.
    """
    assert y_true.shape == y_predicted.shape
    mse = nn.MSELoss(**kwargs)(y_predicted, y_true).item()
    return mse


def auc_score(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Compute AUC-ROC score.
    """
    fpr, tpr, threshold = roc_curve(y_true, y_predicted, **kwargs)
    return auc(fpr, tpr)


def top_k_accuracy_score(y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
    """
    Compute the top-k accuracy score
    in a multi-class setting.

    Conversion to numpy is expensive in this
    case. Stick to using PyTorch tensors.
    """
    k = int(match(kwargs["criterion"], TOP_K_ACCURACY_REGEX))  # Extract k
    _, topk_indices = torch.topk(y_predicted, k, dim=1)  # Compute the top-k predicted classes
    correct_examples = torch.eq(y_true[..., None, ...].long(), topk_indices).any(dim=1)
    top_k_accuracy = correct_examples.float().mean().item()
    return top_k_accuracy


def canonicalize(eval_criterion: str, name: Optional[str] = None, regex: Optional[str] = None) -> str:
    """
    Convert a given `eval_criterion` to its canonical name.
    E.g. for 'top_1_accuracy', it would return 'top_k_accuracy'.
    """
    assert not (name is None and regex is None)
    if name is not None and eval_criterion == name:
        return name
    if regex is not None and match(eval_criterion, regex) is not None:
        return name
    return None


def match(eval_criterion: str, regex: Optional[str] = None) -> str:
    """
    If a regex is provided (i.e., for dynamic criteria),
    try to match to it with the given `eval_criterion`.
    If a match is found, return the (first) group in the result.
    """
    if regex:
        match = re.search(regex, eval_criterion)
        if match:
            return match.group(1)
    return None


class FocalLoss(nn.Module):
    """
    Implement the focal loss for binary classification (ignores regression).
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
        probs1 = torch.sigmoid(outputs[:, 1])
        targets = y.float()

        # alpha balancing weights = alpha if y = 1 else (1-alpha)
        w_alpha = torch.ones_like(targets, device=targets.device) * self.alpha
        w_alpha = torch.where(torch.eq(targets, 1.0), w_alpha, 1.0 - w_alpha)

        # focal weights = (1-p)^gamma if y = 1 else p^gamma
        w_focal = torch.where(torch.eq(targets, 1.0), 1.0 - probs1, probs1)
        w_focal = torch.pow(w_focal, self.gamma)

        # Focal loss = w_alpha * w_focal * BCELoss
        bce_loss = -(targets * probs1.log() + (1.0 - targets) * (1.0 - probs1).log())
        focal_loss = w_alpha * w_focal * bce_loss

        return focal_loss.mean()

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"


# Add your choice of loss criterion here
# See the top for instructions on how
REGRESSION_LOSS_CRITERIA = ["mse"]
CLASSIFICATION_LOSS_CRITERIA = ["cross-entropy", "focal-loss"]
LOSS_CRITERIA = REGRESSION_LOSS_CRITERIA + CLASSIFICATION_LOSS_CRITERIA

# Add your choice of eval criterion here
# See the top for instructions on how
# Note: sklearn metrics seem to be compatible with torch tensors on their own
EVAL_METRIC_FUNCTIONS = {
    "mse": {"eval_fn": get_mse_loss, "model_type": "regression"},
    "accuracy": {
        "preprocess_fn": highest_prob_class,
        "eval_fn": correct_argument_order(accuracy_score),
        "model_type": "classification",
    },
    "precision": {
        "preprocess_fn": highest_prob_class,
        "eval_fn": correct_argument_order(precision_score),
        "model_type": "classification",
    },
    "recall": {
        "preprocess_fn": highest_prob_class,
        "eval_fn": correct_argument_order(recall_score),
        "model_type": "classification",
    },
    "f1": {
        "preprocess_fn": highest_prob_class,
        "eval_fn": correct_argument_order(f1_score),
        "model_type": "classification",
    },
    "auc": {"preprocess_fn": prob_class1, "eval_fn": auc_score, "model_type": "classification"},
    "top_k_accuracy": {
        "eval_fn": top_k_accuracy_score,
        "regex": TOP_K_ACCURACY_REGEX,
        "model_type": "classification",
    },
}


# Wrangling for convenience
for k, v in EVAL_METRIC_FUNCTIONS.items():
    EVAL_METRIC_FUNCTIONS[k]["preprocess_fn"] = v.get("preprocess_fn", dummy_preprocessing)
    EVAL_METRIC_FUNCTIONS[k]["regex"] = v.get("regex", None)
    EVAL_METRIC_FUNCTIONS[k]["model_type"] = v.get("model_type", "classification")

# Wrangling for convenience
PREPROCESSING_FUNCTIONS = {}
for k, v in EVAL_METRIC_FUNCTIONS.items():
    preprocess_fn = v["preprocess_fn"]
    if PREPROCESSING_FUNCTIONS.get(preprocess_fn) is None:
        PREPROCESSING_FUNCTIONS[preprocess_fn] = []
    PREPROCESSING_FUNCTIONS[preprocess_fn].append(k)
