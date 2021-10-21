import unittest

from pytorch_common.additional_configs import BaseModelConfig
from pytorch_common.models import create_model
from pytorch_common.models_dl import (
    MultiLayerClassifier,
    MultiLayerRegressor,
    SingleLayerClassifier,
    SingleLayerRegressor,
)
from pytorch_common.utils import get_trainable_params


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up model config with default hyperparameters.
        """
        cls.default_config_dict = {"in_dim": 8, "h_dim": 4, "num_layers": 2, "num_classes": 2}
        cls.default_config = BaseModelConfig(cls.default_config_dict)

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

    def test_get_trainable_params(self):
        """
        Test computation of trainable and
        total parameters of a model.
        """
        model = create_model("multi_layer_classifier", self.default_config)  # Create model

        self._test_model_parameters(model, 66, 66)  # Full model
        self._test_model_parameters(model.trunk[0], 36, 36)  # First layer
        self._test_model_parameters(model.trunk[1], 20, 20)  # Second layer
        self._test_model_parameters(model.classifier, 10, 10)  # Classifier

    def test_freeze_unfreeze_module(self):
        """
        Test freezing and unfreezing of
        parts of or the entirety of a model.
        """
        model = create_model("multi_layer_classifier", self.default_config)  # Create model
        total = model.num_parameters["total"]  # Get total model parameters

        self._test_model_parameters(model, total, total)  # Full model

        model.freeze_module(model.trunk[0])  # Freeze first layer
        trainable = sum(
            [get_trainable_params(m)["total"] for m in [model.trunk[1], model.classifier]]
        )  # All except first layer
        self._test_model_parameters(model, trainable, total)

        model.freeze_module()  # Freeze full model
        self._test_model_parameters(model, 0, total)

        model.unfreeze_module(model.trunk[1])  # Unfreeze second layer
        trainable = get_trainable_params(model.trunk[1])["total"]
        self._test_model_parameters(model, trainable, total)

        model.unfreeze_module()  # Unfreeze full model
        self._test_model_parameters(model, total, total)

    def _test_model_parameters(self, model, trainable_params, total_params):
        """
        Generic function to ensure the trainable and total
        parameters of a model match those expected.
        """
        parameters = get_trainable_params(model)
        self.assertEqual(parameters["trainable"], trainable_params)
        self.assertEqual(parameters["total"], total_params)


if __name__ == "__main__":
    unittest.main()
