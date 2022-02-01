import unittest

import numpy as np
import torch
from munch import Munch
from sklearn.metrics import accuracy_score, auc, f1_score, mean_squared_error, precision_score, recall_score, roc_curve

from pytorch_common import metrics
from pytorch_common.types import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    _EvalCriterionOrCriteria,
    _Loss,
    _StringDict,
)
from pytorch_common.utils import compare_tensors_or_arrays


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up config with default parameters.
        """
        cls.config = {
            "model_type": "classification",
            "classification_type": "binary",
            "loss_criterion": "cross-entropy",
            "loss_kwargs": {},
            "eval_criteria": ["accuracy"],
            "eval_criteria_kwargs": {},
        }

    def test_loss_eval_criteria(self):
        """
        Test different loss and eval criteria configurations.
        """
        eval_criteria = metrics.EvalCriteria().names

        # Ensure inter-compatibility of all supported
        # loss and eval criteria for binary classification
        for loss_criterion in metrics.LOSS_CRITERIA:
            self._get_loss_eval_criteria({"loss_criterion": loss_criterion, "eval_criteria": eval_criteria})

        # Ensure inter-compatibility of supported loss and
        # eval criteria for multiclass/multilabel classification
        for loss_criterion in metrics.LOSS_CRITERIA:
            for classification_type in ["multiclass", "multilabel"]:
                dictionary = {
                    "classification_type": classification_type,
                    "loss_criterion": loss_criterion,
                    "eval_criteria": eval_criteria,
                }
                if loss_criterion == "focal-loss":
                    # FocalLoss only compatible with binary classification
                    self._test_error(self._get_loss_eval_criteria, ValueError, dictionary)
                else:
                    # Param `multilabel_reduction` required for multilabel classification
                    if classification_type == "multilabel":
                        self._test_error(self._get_loss_eval_criteria, ValueError, dictionary)
                        for key in ["loss_kwargs", "eval_criteria_kwargs"]:
                            dictionary[key] = {"multilabel_reduction": "mean"}
                    self._get_loss_eval_criteria(dictionary)

    def test_regression_metrics(self):
        """
        Test the computation correctness of
        all supported regression metrics.
        """
        predictions, targets = np.array([1.5, 2.5, 5.0]), np.array([1.4, 2.2, 5.1])

        # Compute true values
        true_values = {"mse": mean_squared_error(targets, predictions)}

        # Compute all evaluation criteria
        self._test_metrics(predictions, targets, metrics.EvalCriteria("regression").names, true_values)

    def test_classification_metrics(self):
        """
        Test the computation correctness of
        all supported classification metrics.
        """
        predictions, targets = np.array([[0.2, 0.8], [0.4, 0.6], [1.0, 0.0]]), np.array([0, 1, 0])
        predictions_binary = np.where(predictions[:, 1] > 0.5, 1, 0)  # Binary predictions

        # Compute true values
        true_values = {
            "accuracy": accuracy_score(targets, predictions_binary),
            "precision": precision_score(targets, predictions_binary),
            "recall": recall_score(targets, predictions_binary),
            "f1": f1_score(targets, predictions_binary),
            "top_1_accuracy": accuracy_score(targets, predictions_binary),
            "top_2_accuracy": 1.0,
        }
        fpr, tpr, _ = roc_curve(targets, predictions[:, 1])
        true_values["auc"] = auc(fpr, tpr)

        # Compute all evaluation criteria
        eval_criteria = metrics.EvalCriteria("classification").names + ["top_1_accuracy", "top_2_accuracy"]
        eval_criteria.remove("top_k_accuracy")
        self._test_metrics(predictions, targets, eval_criteria, true_values)

    def _get_loss_eval_criteria(self, dictionary: Dict) -> Tuple[_Loss, _Loss, _EvalCriterionOrCriteria]:
        """
        Load the default config, override it
        with `dictionary`, and get the loss
        and eval criteria with this config.
        """
        config = Munch(self._get_merged_dict(dictionary))
        return metrics.get_loss_eval_criteria(config)

    def _test_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        eval_criteria: List[str],
        true_values: _StringDict,
    ) -> None:
        """
        Test that all `eval_criteria` values computed using
        `predictions` and `targets` match the `true_values`.
        """
        # Get torch tensors
        predictions = torch.as_tensor(predictions).float()
        targets = torch.as_tensor(targets).float()

        # Get all eval metrics
        _, _, eval_criteria = self._get_loss_eval_criteria({"eval_criteria": eval_criteria})
        eval_metrics = eval_criteria(predictions, targets)

        # Test that computed metrics match true values
        for metric, value in eval_metrics.items():
            np.testing.assert_allclose(value, true_values[metric], atol=1e-3)

    def _test_error(self, func: Callable[[Any], None], error=AssertionError, *args, **kwargs) -> None:
        """
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args` and `kwargs`.
        """
        with self.assertRaises(error):
            func(*args, **kwargs)

    def _get_merged_dict(self, dictionary: Optional[Dict] = None) -> Dict:
        """
        Override default config with
        `dictionary` if provided.
        """
        if dictionary is None:
            return self.config.copy()
        return {**self.config, **dictionary}


if __name__ == "__main__":
    unittest.main()
