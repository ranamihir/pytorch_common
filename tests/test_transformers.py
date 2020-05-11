import unittest
from transformers import AutoModel, AutoConfig
from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import is_transformer_model, create_transformer_model
from pytorch_common import utils


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

	def test_SequencePooler(self):
		def get_seq_pooler(model_name):
			if is_transformer_model(model_name):
				model_type = AutoConfig.from_pretrained(model_name).model_type
			else:
				model_type = model_name
			return utils.SequencePooler(model_type)
		self.assertEqual(get_seq_pooler('bert-base-uncased').model_type, 'bert')
		self.assertEqual(get_seq_pooler('distilbert-base-uncased').model_type, 'distilbert')
		self.assertEqual(get_seq_pooler('google/electra-small-discriminator').model_type, 'electra')
		self.assertEqual(get_seq_pooler('canwenxu/BERT-of-Theseus-MNLI').model_type, 'bert')
		self.assertEqual(get_seq_pooler('dummy').model_type, 'default')

if __name__ == '__main__':
	unittest.main()
