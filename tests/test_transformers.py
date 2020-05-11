import unittest
from transformers import AutoModel
from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import is_transformer_model, create_transformer_model


class TestTransformerModels(unittest.TestCase):
	def test_is_transformer_model(self):
		self.assertTrue(is_transformer_model('distilbert-base-uncased'))
		self.assertTrue(is_transformer_model('google/electra-small-discriminator'))
		self.assertTrue(is_transformer_model('canwenxu/BERT-of-Theseus-MNLI'))

		self.assertFalse(is_transformer_model('single_layer_classifier'))
		self.assertFalse(is_transformer_model('multi_layer_classifier'))

	def test_create_transformer_model(self):
		self.assertEqual(create_transformer_model('distilbert-base-uncased', None).config,
						 AutoModel.from_pretrained('distilbert-base-uncased').config)

		with self.assertRaises(AssertionError):
			config = BaseModelConfig({'in_dim': 10, 'num_classes': 2})
			create_transformer_model('dummy', config)

if __name__ == '__main__':
	unittest.main()
