import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePyTorchModel(nn.Module):
    '''
    Generic PyTorch Model implementing useful methods
    '''
    def __init__(self, model_type='classification'):
        super().__init__()

        self.model_type = model_type
        self.__name__ = self.__class__.__name__ # Set class name

    def initialize_model(self, init_weights=False, models_to_init=None):
        '''
        Call this function in the base model class.
        :param init_weights: bool, whether to initialize weights
        :param models_to_init: See `initialize_weights()`

        **NOTE**: If your model is/includes a pretrained model,
                  make sure not to set `models_to_init=True` when
                  calling this function, otherwise the pretrained
                  model will also be reinitialized.
        '''
        self.get_trainable_params() # Print number of trainable parameters
        self.print_model() # Print model architecture
        if init_weights: # Initialize weights
            self.initialize_weights(models_to_init)

    def get_trainable_params(self):
        '''
        Print and return the number of trainable parameters of the model
        '''
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info('Number of trainable/total parameters in {}: {}/{}'.format(self.__name__, \
                      num_trainable_params, num_params))
        return {'trainable': num_trainable_params, 'total': num_params}

    def print_model(self):
        '''
        Print model architecture
        '''
        logging.info(self)

    def initialize_weights(self, models_to_init=None):
        '''
        If `models_to_init` is provided, it will only
        initialize their weights, otherwise whole model's.
        :param models_to_init: Model(s) to be initialized.
                               Can take the following values:
                               - None, will initialize self
                               - list or object of type
                                 nn.Module/BasePyTorchModel,
                                 will initialize its weights
        '''
        if models_to_init is None:
            models_to_init = self # Base model
        if not isinstance(models_to_init, list):
            models_to_init = [models_to_init]
        for model in models_to_init:
            self._initialize_weights_for_one_model(model)

    def _initialize_weights_for_one_model(self, model):
        '''
        Initialize weights for all Conv2d, BatchNorm2d, Linear, and Embedding layers
        # TODO: Improve init schemes / params
        '''
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
    def predict_proba(self, outputs, threshold=None):
        '''
        Returns predicted labels and probabilities
        for `model_type = 'classification'`.
        :param outputs: Raw model output logits
        :param threshold: Threshold probability to compute
                          hard label predictions.
        :return preds: Predicted classes
        :return probs: Predicted probabilities of each class
        '''
        if self.model_type != 'classification' and threshold is not None:
            raise ValueError(f'Param "threshold" ("{threshold}") can only '
                             f'be provided for classification models.')

        probs = F.softmax(outputs, dim=-1) # Get probabilities of each class
        n_classes = probs.shape[-1]

        if threshold is not None:
            device = probs.device
            pos_tensor, neg_tensor = torch.tensor(1, device=device), torch.tensor(0, device=device)
            labels_for_class_i = lambda i: torch.where(probs[...,i] >= threshold, pos_tensor, neg_tensor)
            if n_classes == 2: # Only get labels for class 1 if binary classification
                preds = labels_for_class_i(1)
            else: # Get labels for each class if multiclass classification
                preds = torch.stack([labels_for_class_i(i) for i in range(n_classes)], dim=1).max(dim=-1)[1]
        else:
            # Get class with max probability (same as threshold=0.5)
            preds = probs.max(dim=-1)[1]

        if n_classes == 2: # Only get probs for class 1 if binary classification
            probs = probs[...,1]
        return preds, probs

    def freeze_module(self, model_to_freeze=None):
        '''
        Freeze a given `model_to_freeze`, i.e.,
        all its children are gradient free.
        '''
        if model_to_freeze is None:
            model_to_freeze = self # Base model

        # Extract model name from class if not present already (for `transformers` models)
        model_name = getattr(model_to_freeze, '__name__', model_to_freeze.__class__.__name__)

        # Perform freezing
        logging.info(f'Freezing {model_name}...')
        for child in model_to_freeze.children():
            for param in child.parameters():
                param.requires_grad = False
        logging.info('Done.')
        self.get_trainable_params()


class SingleLayerClassifier(BasePyTorchModel):
    '''
    Dummy Single-layer multi-class "neural" network.
    '''
    def __init__(self, config):
        super().__init__(model_type=config.model_type)
        self.in_dim = config.in_dim
        self.n_classes = config.n_classes

        self.fc = nn.Linear(config.in_dim, config.n_classes)

        self.initialize_model()

    def forward(self, x):
        return self.fc(x)


class MultiLayerClassifier(BasePyTorchModel):
    '''
    Dummy Multi-layer multi-class model.
    '''
    def __init__(self, config):
        super().__init__(model_type=config.model_type)
        self.in_dim = config.in_dim
        self.h_dim = config.h_dim
        self.n_classes = config.n_classes
        self.n_layers = config.n_layers

        trunk = [nn.Sequential(nn.Linear(self.in_dim, self.h_dim), \
                               nn.ReLU(inplace=True))]
        for _ in range(self.n_layers-1):
            layer = nn.Sequential(
                nn.Linear(self.h_dim, self.h_dim),
                nn.ReLU(inplace=True)
            )
            trunk.append(layer)
        self.trunk = nn.Sequential(*trunk)

        self.classifier = nn.Linear(self.h_dim, self.n_classes)

        self.initialize_model(init_weights=True)

    def forward(self, x):
        return self.classifier(self.trunk(x))
