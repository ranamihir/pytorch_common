import unittest

from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model, is_transformer_model, create_transformer_model
from pytorch_common import utils

# Check if `transformers` installed
try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_INSTALLED = True
except ImportError:
    TRANSFORMERS_INSTALLED = False


class TestTransformerModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        '''
        Skip all of the following tests if
        `transformers` is not installed.
        '''
        if not TRANSFORMERS_INSTALLED:
            raise unittest.SkipTest('`transformers` not installed.')

    def test_is_transformer_model(self):
        '''
        Test identification of `transformers`-based models.
        '''
        self.assertTrue(is_transformer_model('distilbert-base-uncased'))
        self.assertTrue(is_transformer_model('google/electra-small-discriminator'))
        self.assertTrue(is_transformer_model('canwenxu/BERT-of-Theseus-MNLI'))

        self.assertFalse(is_transformer_model('single_layer_classifier'))
        self.assertFalse(is_transformer_model('multi_layer_classifier'))
        self.assertFalse(is_transformer_model('single_layer_regressor'))
        self.assertFalse(is_transformer_model('multi_layer_regressor'))

    def test_create_transformer_model(self):
        '''
        Test creation of `transformers`-based models.
        '''
        # Test that correct `transformers` model is
        # loaded by ensuring that configs match
        self.assertEqual(create_model('distilbert-base-uncased', None).config,
                         AutoModel.from_pretrained('distilbert-base-uncased').config)
        self.assertEqual(create_transformer_model('distilbert-base-uncased', None).config,
                         AutoModel.from_pretrained('distilbert-base-uncased').config)

        # Check that errror is raised for wrong model
        with self.assertRaises(AssertionError):
            config = BaseModelConfig({'in_dim': 10, 'num_classes': 2})
            create_transformer_model('dummy', config)

        # Check that errror is raised for wrong model
        with self.assertRaises(RuntimeError):
            config = BaseModelConfig({'in_dim': 10, 'num_classes': 2})
            create_model('dummy', config)

    def test_sequence_pooler(self):
        '''
        Test `SequencePooler` setup for different configurations.
        '''
        def get_seq_pooler(model_name):
            '''
            Return the `SequencePooler` for `model_name`.
            '''
            if is_transformer_model(model_name):
                model_type = AutoConfig.from_pretrained(model_name).model_type
            else: # Pass `model_name` as-is if not `transformers`-based model
                model_type = model_name
            return utils.SequencePooler(model_type)

        # Test different model configurations
        self.assertEqual(get_seq_pooler('bert-base-uncased').model_type, 'bert')
        self.assertEqual(get_seq_pooler('distilbert-base-uncased').model_type, 'distilbert')
        self.assertEqual(get_seq_pooler('google/electra-small-discriminator').model_type, 'electra')
        self.assertEqual(get_seq_pooler('canwenxu/BERT-of-Theseus-MNLI').model_type, 'bert')
        self.assertEqual(get_seq_pooler('dummy').model_type, 'default')


if __name__ == '__main__':
    unittest.main()
