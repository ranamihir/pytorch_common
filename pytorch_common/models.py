from vrdscommon.dsp_pipeline import DspPipeline
import logging
from .models_dl import BasePyTorchModel, SingleLayerClassifier, MultiLayerClassifier


def create_model(model_name, config):
    if model_name == 'base_pytorch_model':
        model = BasePyTorchModel(config.model_type)
    elif model_name == 'single_layer_classifier':
        model = SingleLayerClassifier(config)
    elif model_name == 'multi_layer_classifier':
        model = MultiLayerClassifier(config)
    else:
        raise RuntimeError(f'Unknown model name {model_name}.')
    return model

def create_estimator(estimator_name, config):
    return create_model(estimator_name, config)
