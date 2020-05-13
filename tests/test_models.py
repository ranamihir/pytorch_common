import unittest

from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model
from pytorch_common.models_dl import SingleLayerClassifier, MultiLayerClassifier


class TestModels(unittest.TestCase):
    def test_create_model(self):
        '''
        Test creation of non-transformer
        based supported models.
        '''
        config = BaseModelConfig({'in_dim': 10, 'num_classes': 2})
        self.assertIsInstance(create_model('single_layer_classifier', config), SingleLayerClassifier)

        config = BaseModelConfig({'in_dim': 10, 'h_dim': 10, 'num_layers': 1, 'num_classes': 2})
        self.assertIsInstance(create_model('multi_layer_classifier', config), MultiLayerClassifier)


if __name__ == '__main__':
    unittest.main()
