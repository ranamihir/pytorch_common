import unittest

from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model
from pytorch_common.models_dl import (
    SingleLayerClassifier, MultiLayerClassifier, SingleLayerRegressor, MultiLayerRegressor
)


class TestModels(unittest.TestCase):
    def test_create_model(self):
        """
        Test creation of non-transformer
        based supported models.
        """
        in_dim, h_dim, out_dim, num_layers, num_classes = 8, 4, 2, 2, 2

        base_classifier_config = {"in_dim": in_dim, "num_classes": num_classes}
        base_regressor_config = {"in_dim": in_dim, "out_dim": out_dim}
        multi_layer_config = {"h_dim": h_dim, "num_layers": num_layers}

        config = BaseModelConfig(base_classifier_config)
        self.assertIsInstance(create_model("single_layer_classifier", config), SingleLayerClassifier)

        config = BaseModelConfig({**base_classifier_config, **multi_layer_config})
        self.assertIsInstance(create_model("multi_layer_classifier", config), MultiLayerClassifier)

        config = BaseModelConfig(base_regressor_config)
        self.assertIsInstance(create_model("single_layer_regressor", config), SingleLayerRegressor)

        config = BaseModelConfig({**base_regressor_config, **multi_layer_config})
        self.assertIsInstance(create_model("multi_layer_regressor", config), MultiLayerRegressor)


if __name__ == "__main__":
    unittest.main()
