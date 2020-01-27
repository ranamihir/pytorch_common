import unittest
import pandas as pd

from pytorch_common.models import NoOpTransformer


class TransformersTest(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({'a': [1, 2, 3], 'b': list('def'), 'c': [True, False, True]})

    def test_no_op_transformer(self):
        transformer = NoOpTransformer()
        transformed_df = transformer.transform(self.test_data)
        self.assertTrue(self.test_data.equals(transformed_df))


if __name__ == '__main__':
    unittest.main()
