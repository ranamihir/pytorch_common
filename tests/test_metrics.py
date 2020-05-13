import unittest
from munch import Munch

from pytorch_common.metrics import LOSS_CRITERIA, EVAL_CRITERIA, get_loss_eval_criteria


class TestMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Set up config with default parameters.
        '''
        cls.config = {
            'model_type': 'classification',
            'classification_type': 'binary',
            'loss_criterion': 'cross-entropy',
            'loss_kwargs': {},
            'eval_criteria': ['accuracy'],
            'eval_criteria_kwargs': {}
        }

    def test_loss_eval_criteria(self):
        '''
        Test different loss and eval criteria configurations.
        '''
        # Ensure inter-compatibility of all supported
        # loss and eval criteria for binary classification
        for loss_criterion in LOSS_CRITERIA:
            self._get_loss_eval_criteria({'loss_criterion': loss_criterion,
                                          'eval_criteria': EVAL_CRITERIA})

        # Ensure inter-compatibility of supported loss and
        # eval criteria for multiclass/multilabel classification
        for loss_criterion in LOSS_CRITERIA:
            for classification_type in ['multiclass', 'multilabel']:
                dictionary = {'classification_type': classification_type,
                              'loss_criterion': loss_criterion, 'eval_criteria': EVAL_CRITERIA}
                if loss_criterion == 'focal-loss':
                    # FocalLoss only compatible with binary classification
                    self._test_error(self._get_loss_eval_criteria, dictionary, error=ValueError)
                else:
                    # Param `multilabel_reduction` required for multilabel classification
                    if classification_type == 'multilabel':
                        self._test_error(self._get_loss_eval_criteria, dictionary, error=ValueError)
                        for key in ['loss_kwargs', 'eval_criteria_kwargs']:
                            dictionary[key] = {'multilabel_reduction': 'mean'}
                    self._get_loss_eval_criteria(dictionary)

    def _get_loss_eval_criteria(self, dictionary):
        '''
        Load the default config, override it with
        `dictionary`, and get the loss and eval
        criteria with this config.
        '''
        config = Munch(self._get_merged_dict(dictionary))
        return get_loss_eval_criteria(config)

    def _test_error(self, func, args, error=AssertionError):
        '''
        Generic code to assert that `error`
        is raised when calling a function
        `func` with arguments `args`.
        '''
        with self.assertRaises(error):
            func(args)

    def _get_merged_dict(self, dictionary=None):
        '''
        Override default config with
        `dictionary` if provided.
        '''
        if dictionary is None:
            return self.config.copy()
        return {**self.config, **dictionary}


if __name__ == '__main__':
    unittest.main()
