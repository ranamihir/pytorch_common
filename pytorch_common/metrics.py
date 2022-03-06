"""
To add a new loss / evaluation criterion, please see the instructions in metric_utils.py
"""
import numpy as np
import torch
import torch.nn as nn

from .metric_utils import (
    EVAL_METRIC_FUNCTIONS,
    LOSS_CRITERIA,
    LOSS_REDUCTIONS,
    PREPROCESSING_FUNCTIONS,
    FocalLoss,
    canonicalize,
    match,
    top_k_accuracy_scores,
)
from .types import *


class EvalCriterion:
    """
    Base class for computation of all supported evaluation criteria.

    It handles:
      - Static criteria like `"accuracy"`, `"precision"`, `"mse"`, etc.
      - Dynamic criteria like `"top_1_accuracy"`, `"top_10_accuracy"`, etc. (on the fly)
    """

    def __init__(
        self,
        criterion: Optional[str] = None,
        eval_fn: Optional[Callable[[_TensorOrArray, _TensorOrArray], float]] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        """
        :param criterion: Criterion name in config, e.g. `"top_5_accuracy"`
        :param eval_fn: Metric evaluating function
        :param name: Canonical name, e.g. `"top_k_accuracy"`
        """
        self.criterion = criterion
        self.eval_fn = eval_fn
        self._name = name
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs) -> float:
        """
        Used in `EvalCriteria`.
        """
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(criterion={self.criterion}, name={self.name})"

    def __eq__(self, eval_criterion: str) -> bool:
        """
        Used in `EvalCriteria.__contains__()`
        for determining if a given `eval_criterion`
        is supported or not.
        """
        return self.canonicalize(eval_criterion) is not None

    def canonicalize(self, eval_criterion: str) -> bool:
        """
        Convert a given `eval_criterion` to its canonical name.
        E.g. for `"top_1`_accuracy"`, it would return `"top_k_accuracy"`.
        """
        return canonicalize(eval_criterion, self.name, self.regex)

    def match(self, eval_criterion: str) -> Optional[str]:
        """
        If a regex is provided (i.e., for dynamic criteria),
        try to match to it with the given `eval_criterion`.
        If a match is found, return the (first) group in the result.
        """
        return match(eval_criterion, self.regex)

    @property
    def name(self):
        return self._name if self._name is not None else self.canonicalize(self.criterion)

    @property
    def regex(self):
        return EVAL_METRIC_FUNCTIONS[self.name]["regex"]

    def compute(self, y_predicted: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        This function does the actual
        evaluation metric computation.

        NOTE: The inputs to this function
        must be preprocessed. E.g. for accuracy,
        `y_predicted` should already been hard
        predictions rather than raw logits per class.

        Used in `__call__`.
        """
        # Pass criterion for parsing dynamic criteria.
        # E.g.: For "top_5_accuracy", k=5 will be deduced
        #       on the fly based on `self.criterion`.
        if self.regex:
            self.kwargs["criterion"] = self.criterion
        return self.eval_fn(y_predicted, y_true, **self.kwargs)


class EvalCriteria:
    """
    Collates all supported criteria for regression and classification.

    Creates all `EvalCriterion` objects for
    the given list of `eval_criteria` strings.

    The idea is that once defined, all that's required
    is to call an instance of this class, and that would
    compute all evaluation metrics automatically in an
    optimized manner.
    """

    REGRESSION_EVAL_CRITERIA = [k for k, v in EVAL_METRIC_FUNCTIONS.items() if v["model_type"] == "regression"]
    CLASSIFICATION_EVAL_CRITERIA = [k for k, v in EVAL_METRIC_FUNCTIONS.items() if v["model_type"] == "classification"]
    ALL_EVAL_CRITERIA = REGRESSION_EVAL_CRITERIA + CLASSIFICATION_EVAL_CRITERIA

    initialize = lambda _list: [EvalCriterion(name=name) for name in _list]
    REGRESSION = initialize(REGRESSION_EVAL_CRITERIA)
    CLASSIFICATION = initialize(CLASSIFICATION_EVAL_CRITERIA)
    ALL = initialize(ALL_EVAL_CRITERIA)

    def __init__(
        self,
        model_type: Optional[str] = None,
        classification_type: Optional[str] = None,
        eval_criteria: Optional[List[str]] = None,
        **eval_criteria_kwargs: _StringDict,
    ):
        self.model_type = model_type
        self.classification_type = classification_type
        self.eval_criteria_kwargs = eval_criteria_kwargs
        self._criteria = []
        self.init_eval_criteria(eval_criteria)

        self.check_multilabel()
        self.create_all_eval_metrics()

    def __call__(self, *args, **kwargs) -> float:
        """
        Called in `train_utils.perform_one_epoch()`
        for all metric computation.
        """
        return self.compute(*args, **kwargs)

    def __repr__(self) -> str:
        return ", ".join([str(criterion) for criterion in self.criteria])

    def __iter__(self):
        """
        Useful for iterating over
        all eval criteria.
        """
        return iter(self.criteria)

    def __getitem__(self, index: int) -> EvalCriterion:
        """
        Useful for iterating over
        all eval criteria.
        """
        return self.criteria[index]

    def __contains__(self, eval_criterion: str) -> bool:
        """
        Returns true of `eval_criterion` is
        present in list of supported criteria.

        NOTE that `==` works because of the defined
        `__eq__` in `EvalCriterion`.
        """
        return any(eval_criterion == supported_criterion for supported_criterion in self.criteria)

    @property
    def names(self):
        """
        List of names of supported criteria.
        """
        return self.eval_criteria

    @property
    def criteria(self):
        """
        Get the list of specified
        `EvalCriterion` objects.
        """
        return self._criteria

    def init_eval_criteria(self, eval_criteria: Optional[List[str]] = None) -> None:
        """
        Define list of evaluation criteria strings
        based on the provided list / model type.
        """
        if eval_criteria is not None:
            self.eval_criteria = eval_criteria
        elif self.model_type is None:
            self.eval_criteria = self.ALL_EVAL_CRITERIA
        elif self.model_type == "classification":
            self.eval_criteria = self.CLASSIFICATION_EVAL_CRITERIA
        elif self.model_type == "regression":
            self.eval_criteria = self.REGRESSION_EVAL_CRITERIA
        else:
            raise ValueError

    def compute(self, y_predicted: torch.Tensor, y_true: torch.Tensor, **kwargs) -> float:
        """
        Returns a dict of names and computed evaluation
        metric values for all specified criteria.

        It is optimized to share the computation all
        criteria that have the same preprocessing step.
        """
        if self.is_multilabel:
            # Compute results for each class
            results_per_class = [self.compute_per_class(y_predicted, y_true[..., i]) for i in range(y_true.shape[-1])]

            # Aggregate results across all classes
            results = {}
            for criterion in results_per_class[0]:
                results[criterion] = self.agg_func([results_class[criterion] for results_class in results_per_class])
        else:
            # Compute results directly if regression / not multilabel
            results = self.compute_per_class(y_predicted, y_true)

        return results

    def compute_per_class(self, y_predicted: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Compute function for one class (or overall for regression problems).

        Optimized so that computation is shared for metrics
        that have a common preprocessing step.
        E.g. accuracy, precision, etc. all compute the
        highest probability class first before the actual
        score computation.
        """
        results = {}
        for preprocess_fn, supported_metrics in PREPROCESSING_FUNCTIONS.items():
            metrics_in_group = [metric_name for metric_name in supported_metrics if metric_name in self.criteria]
            if metrics_in_group:
                preprocessed_input = preprocess_fn(y_predicted, y_true)  # Share preprocessing
                for metric_name in metrics_in_group:
                    # Get eval metrics to be computed in this group
                    eval_metrics = [criterion for criterion in self.criteria if criterion.name == metric_name]

                    # Separate out all top_k_accuracy-based metrics
                    top_k_accuracy_metrics, other_metrics = [], []
                    for eval_metric in eval_metrics:
                        top_k_accuracy_metrics.append(
                            eval_metric
                        ) if eval_metric.name == "top_k_accuracy" else other_metrics.append(eval_metric)

                    # Compute all top_k_accuracy-based metrics together (for efficiency)
                    for eval_metric in top_k_accuracy_metrics:
                        top_k_accuracy_scores(top_k_accuracy_metrics, results, *preprocessed_input)

                    # Compute all other metrics as normal
                    for eval_metric in other_metrics:
                        results[eval_metric.criterion] = eval_metric(*preprocessed_input)
        return results

    def canonicalize(self, eval_criterion: str) -> bool:
        """
        Return the canonical name of `eval_criterion`
        if supported, otherwise None.

        E.g. `"top_1_accuracy"` would be
        canonicalized to `"top_k_accuracy"`.
        """
        for criterion in self.criteria:
            canonical_name = criterion.canonicalize(eval_criterion)
            if canonical_name is not None:
                return canonical_name
        return None

    def get_criterion(self, eval_criterion: str) -> bool:
        """
        Get the `EvalCriterion` object
        for a specified `eval_criterion`.
        """
        criteria = [e for e in self.criteria if e.name == eval_criterion]
        assert len(criteria) <= 1
        return criteria[0] if len(criteria) else None

    def check_multilabel(self):
        """
        Check if it's a multilabel setting.
        If so, define the reduction and
        aggregation functions.
        """
        self.is_multilabel = self.model_type == "classification" and self.classification_type == "multilabel"

        if self.is_multilabel:
            if not self.eval_criteria_kwargs.get("multilabel_reduction"):
                raise ValueError("Param 'multilabel_reduction' must be provided.")
            self.multilabel_reduction = self.eval_criteria_kwargs["multilabel_reduction"]
            if self.multilabel_reduction == "none":
                self.agg_func = np.array
            elif self.multilabel_reduction == "mean":
                self.agg_func = np.mean
            else:
                raise ValueError(
                    f"Param 'multilabel_reduction' ('{self.multilabel_reduction}') must be one of ['mean', 'none']."
                )

    def create_all_eval_metrics(self) -> None:
        """
        Create and store all `EvalCriterion` objects.
        """
        for criterion in self.eval_criteria:
            criterion_kwargs = self.eval_criteria_kwargs.get(criterion, {})
            eval_metric = self.create_eval_metric(criterion, **criterion_kwargs)
            self.criteria.append(eval_metric)

    def create_eval_metric(self, eval_criterion: str, **kwargs) -> None:
        """
        Get the `EvalCriterion` object for a specified `criterion`.

        :param kwargs: Misc kwargs for the eval criterion.
                       Mostly used in multiclass settings. E.g.:
                       - `average` for f1, precision, recall
                       - `pos_label` for auc
                       If it's a multilabel setting,
                       `multilabel_reduction` must be provided:
                        Type of multilabel_reduction to be
                        performed on the list of metric values
                        for each class.
                        Choices: `"sum"` | `"mean"`
        """
        try:
            canonical_name = None
            for name in self.ALL_EVAL_CRITERIA:
                canonical_name = canonicalize(eval_criterion, name, EVAL_METRIC_FUNCTIONS[name]["regex"])
                if canonical_name is not None:
                    break
            eval_fn = EVAL_METRIC_FUNCTIONS[canonical_name]["eval_fn"]
            return EvalCriterion(eval_criterion, eval_fn, canonical_name, **kwargs)
        except KeyError:
            raise ValueError(f"Param 'eval_criterion' ('{eval_criterion}') must be one of {self.names}.")


def get_loss_eval_criteria(config: _Config) -> Tuple[_Loss, _Loss, _EvalCriterionOrCriteria]:
    """
    Define train and val loss and evaluation criteria.
    """
    train_loss_kwargs, val_loss_kwargs = config.loss_kwargs.copy(), config.loss_kwargs.copy()

    # Check correct params for sample weighting
    error_message = (
        "The `reduction` ('{reduction}') for {phase} loss criterion must be "
        "one of {allowed_reductions} if you want to use sample weighting."
    )
    if config.sample_weighting_train:
        assert config.loss_kwargs["reduction_train"] in LOSS_REDUCTIONS, error_message.format(
            reduction=config.loss_kwargs.get("reduction_train"), phase="training", allowed_reductions=LOSS_REDUCTIONS
        )
    if config.sample_weighting_eval:
        assert config.loss_kwargs["reduction_val"] in LOSS_REDUCTIONS, error_message.format(
            reduction=config.loss_kwargs.get("reduction_val"), phase="evaluation", allowed_reductions=LOSS_REDUCTIONS
        )

    # Add / update train loss reduction and get criterion
    reduction_train = train_loss_kwargs.pop("reduction_train", "mean")
    train_loss_kwargs["reduction"] = "none" if config.sample_weighting_train else reduction_train
    train_loss_kwargs.pop("reduction_val", None)
    loss_criterion_train = get_loss_criterion(config, criterion=config.loss_criterion, **train_loss_kwargs)

    # Add / update val loss reduction and get criterion
    reduction_val = val_loss_kwargs.pop("reduction_val", "mean")
    val_loss_kwargs["reduction"] = "none" if config.sample_weighting_eval else reduction_val
    val_loss_kwargs.pop("reduction_train", None)
    loss_criterion_val = get_loss_criterion(config, criterion=config.loss_criterion, **val_loss_kwargs)

    eval_criteria = get_eval_criteria(config, config.eval_criteria, **config.eval_criteria_kwargs)
    return loss_criterion_train, loss_criterion_val, eval_criteria


def get_loss_criterion(config: _Config, criterion: Optional[str] = "cross-entropy", **kwargs) -> _Loss:
    """
    Get the loss criterion function.
    """
    loss_criterion = get_loss_criterion_function(config, criterion=criterion, **kwargs)
    return loss_criterion


def get_eval_criteria(config: _Config, eval_criteria: List[str], **kwargs) -> EvalCriteria:
    """
    Get the `EvalCriteria` object (comprising all eval
    criteria definitions) based on the specified list.
    """
    eval_criteria = EvalCriteria(
        model_type=config.model_type,
        classification_type=config.classification_type,
        eval_criteria=eval_criteria,
        **kwargs,
    )
    return eval_criteria


def get_loss_criterion_function(config: _Config, criterion: Optional[str] = "cross-entropy", **kwargs) -> _Loss:
    """
    Get the function for a given loss `criterion`.

    :param kwargs: Misc kwargs for the loss. E.g.:
                   - `dim` for CrossEntropyLoss
                   - `alpha` and `gamma` for FocalLoss.
                   If it's a multilabel setting,
                   `multilabel_reduction` must be provided:
                    Type of multilabel reduction to be
                    performed on the list of losses for
                    each class.
                    Choices: `"sum"` | `"mean"`
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
                    f"Param 'multilabel_reduction' ('{multilabel_reduction}') must be one of {LOSS_REDUCTIONS}."
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
        return lambda output_hist, y_hist: agg_func(
            torch.stack(
                [loss_criterion(output_hist, y_hist[..., i]) for i in range(y_hist.shape[-1])],
                dim=0,
            )
        )
