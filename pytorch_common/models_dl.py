import logging
from copy import deepcopy
from typing import List, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .additional_configs import BaseModelConfig


_model_or_models = Union[nn.Module, List[nn.Module]]


class BasePyTorchModel(nn.Module):
    """
    Generic PyTorch Model implementing useful methods.
    """
    def __init__(self, model_type: Optional[str] = "classification"):
        super().__init__()

        self.model_type = model_type
        self.__name__ = self.__class__.__name__ # Set model name

    def initialize_model(
        self,
        init_weights: Optional[str] = False,
        models_to_init: Optional[_model_or_models] = None
    ) -> None:
        """
        Call this function in the base model class.
        :param init_weights: bool, whether to initialize weights
        :param models_to_init: See `initialize_weights()`

        **NOTE**: If your model is/includes a pretrained model,
                  make sure to set `init_weights=False` when
                  calling this function, otherwise the pretrained
                  model will also be reinitialized.
        """
        self.print() # Print model architecture
        self.get_trainable_params() # Print number of trainable parameters
        if init_weights: # Initialize weights
            logging.warning(
                "You have set `init_weights=True`. Make sure your model does not include "
                "a pretrained model, otherwise its weights will also be reinitialized."
            )
            self.initialize_weights(models_to_init)

    def get_trainable_params(self) -> Dict[str, int]:
        """
        Print and return the number of trainable parameters of the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info("Number of trainable/total parameters in {}: {}/{}"\
                     .format(self.__name__, num_trainable_params, num_params))
        return {"trainable": num_trainable_params, "total": num_params}

    def print(self) -> None:
        """
        Print the model architecture.
        """
        logging.info(self)

    def initialize_weights(self, models_to_init: Optional[_model_or_models] = None) -> None:
        """
        If `models_to_init` is provided, it will only
        initialize their weights, otherwise whole model's.
        :param models_to_init: Model(s) to be initialized.
                               Can take the following values:
                               - None, will initialize self
                               - list or object of type
                                 nn.Module/BasePyTorchModel,
                                 will initialize their weights
        """
        if models_to_init is None:
            models_to_init = self # Base model
        if not isinstance(models_to_init, list):
            models_to_init = [models_to_init]
        for model in models_to_init:
            self._initialize_weights_for_one_model(model)

    def _initialize_weights_for_one_model(self, model: nn.Module) -> None:
        """
        Initialize weights for all Conv2d, BatchNorm2d, Linear, and Embedding layers.
        # TODO: Improve init schemes / params
        """
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.orthogonal_(m.weight.data)

    @torch.no_grad()
    def predict_proba(self, outputs: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
        """
        Returns predicted labels and probabilities
        for `model_type = "classification"`.
        :param outputs: Raw model output logits
        :param threshold: Threshold probability to compute
                          hard label predictions.
        :return preds: Predicted classes
        :return probs: Predicted probabilities of each class
        """
        if self.model_type != "classification" and threshold is not None:
            raise ValueError(
                f"Param 'threshold' ('{threshold}') can only "
                f"be provided for classification models."
            )

        probs = F.softmax(outputs, dim=-1) # Get probabilities of each class
        num_classes = probs.shape[-1]

        if threshold is not None:
            device = probs.device
            pos_tensor = torch.as_tensor(1, device=device)
            neg_tensor = torch.as_tensor(0, device=device)
            labels_for_class_i = lambda i: torch.where(probs[...,i] >= threshold,
                                                       pos_tensor, neg_tensor)
            if num_classes == 2: # Only get labels for class 1 if binary classification
                preds = labels_for_class_i(1)
            else: # Get labels for each class if multiclass classification
                preds = torch.stack([labels_for_class_i(i) for i in range(num_classes)],
                                    dim=1).max(dim=-1)[1]
        else:
            # Get class with max probability (same as threshold=0.5)
            preds = probs.max(dim=-1)[1]

        if num_classes == 2: # Only get probs for class 1 if binary classification
            probs = probs[...,1]
        return preds, probs

    def copy(self) -> nn.Module:
        """
        Return a copy of the model.
        """
        return deepcopy(self)

    def freeze_module(self, models_to_freeze: Optional[_model_or_models] = None) -> None:
        """
        Freeze the given `models_to_freeze`, i.e.,
        all their children are gradient free.
        """
        self._change_frozen_state(models_to_freeze, freeze=True)

    def unfreeze_module(self, models_to_unfreeze: Optional[_model_or_models] = None) -> None:
        """
        Unfreeze the given `models_to_unfreeze`, i.e.,
        all their children have gradients.
        """
        self._change_frozen_state(models_to_unfreeze, freeze=False)

    def _change_frozen_state(
        self,
        models: Optional[_model_or_models] = None,
        freeze: Optional[bool] = True
    ) -> None:
        """
        Freeze or unfreeze the given `models`, i.e.,
        all their children will / won't have gradients.
        :param models: Models / modules to freeze / unfreeze
                       Can take the following values:
                       - None, will alter self
                       - list or object of type
                       - nn.Module/BasePyTorchModel,
                         will alter their state
        :freeze: Whether to freeze or unfreeze (bool)
        """
        if models is None:
            models = self # Base model
        if not isinstance(models, list):
            models = [models]
        for model in models:
            self._change_frozen_state_for_one_model(model, freeze)
        self.get_trainable_params()

    def _change_frozen_state_for_one_model(
        self, model: Optional[nn.Module] = None,
        freeze: Optional[bool] = True
    ) -> None:
        """
        Freeze or unfreeze a given `model`, i.e.,
        all their children will / won't have gradients.
        :param model: Model / module to freeze / unfreeze
                      Can take the following values:
                      - None, will alter self
                      - list or object of type
                      - nn.Module/BasePyTorchModel,
                        will alter their state
        :freeze: Whether to freeze or unfreeze (bool)
        """
        # Extract model name from class if not present already (for `transformers` models)
        model_name = getattr(model, "__name__", model.__class__.__name__)

        logging.info("{} {}...".format("Freezing" if freeze else "Unfreezing", model_name))
        for param in model.parameters():
            param.requires_grad = not freeze
        logging.info("Done.")


class SingleLayerClassifier(BasePyTorchModel):
    """
    Dummy single-layer multi-class "neural" network.
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__(model_type="classification")
        self.in_dim = config.in_dim
        self.num_classes = config.num_classes

        self.fc = nn.Linear(self.in_dim, self.num_classes)

        self.initialize_model(init_weights=True)

    def forward(self, x):
        return self.fc(x)


class MultiLayerClassifier(BasePyTorchModel):
    """
    Dummy multi-layer multi-class neural network.
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__(model_type="classification")
        self.in_dim = config.in_dim
        self.h_dim = config.h_dim
        self.num_classes = config.num_classes
        self.num_layers = config.num_layers

        trunk = [nn.Sequential(nn.Linear(self.in_dim, self.h_dim), nn.ReLU(inplace=True))]
        for _ in range(self.num_layers-1):
            layer = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(inplace=True)
            )
            trunk.append(layer)
        self.trunk = nn.Sequential(*trunk)

        self.classifier = nn.Linear(self.h_dim, self.num_classes)

        self.initialize_model(init_weights=True)

    def forward(self, x):
        return self.classifier(self.trunk(x))


class SingleLayerRegressor(BasePyTorchModel):
    """
    Dummy single-layer "neural" regressor.
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__(model_type="regression")
        self.in_dim = config.in_dim
        self.out_dim = config.out_dim

        self.fc = nn.Linear(self.in_dim, self.out_dim)

        self.initialize_model(init_weights=True)

    def forward(self, x):
        return self.fc(x)


class MultiLayerRegressor(BasePyTorchModel):
    """
    Dummy multi-layer neural regressor.
    """
    def __init__(self, config: BaseModelConfig):
        super().__init__(model_type="regression")
        self.in_dim = config.in_dim
        self.h_dim = config.h_dim
        self.out_dim = config.out_dim
        self.num_layers = config.num_layers

        trunk = [nn.Sequential(nn.Linear(self.in_dim, self.h_dim), nn.ReLU(inplace=True))]
        for _ in range(self.num_layers-1):
            layer = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(inplace=True)
            )
            trunk.append(layer)
        self.trunk = nn.Sequential(*trunk)

        self.head = nn.Linear(self.h_dim, self.out_dim)

        self.initialize_model(init_weights=True)

    def forward(self, x):
        return self.head(self.trunk(x))
