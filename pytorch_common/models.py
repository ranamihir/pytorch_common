from vrdscommon.dsp_pipeline import DspPipeline

import numpy as np
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import send_model_to_device
from .models_dl import BasePyTorchModel, SingleLayerClassifier, MultiLayerClassifier

def create_model(model_name, config, send_to_device=True):
    if model_name == 'base_pytorch_model':
        model = BasePyTorchModel(config.model_type)
    elif model_name == 'single_layer_classifier':
        model = SingleLayerClassifier(config)
    elif model_name == 'multi_layer_classifier':
        model = MultiLayerClassifier(config)
    else:
        raise RuntimeError('Unknown model name {}.'.format(model_name))

    if send_to_device:
        return send_model_to_device(model, config.device, config.device_ids)
    return model
