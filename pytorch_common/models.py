import torch.nn as nn

from .additional_configs import BaseModelConfig
from .models_dl import MultiLayerClassifier, MultiLayerRegressor, SingleLayerClassifier, SingleLayerRegressor
from .types import Optional
from .utils import get_file_path


def create_model(model_name: str, config: Optional[BaseModelConfig] = None) -> nn.Module:
    """
    Create and return the appropriate
    model from the provided config.
    """
    if model_name == "single_layer_classifier":
        model = SingleLayerClassifier(config)
    elif model_name == "multi_layer_classifier":
        model = MultiLayerClassifier(config)
    elif model_name == "single_layer_regressor":
        model = SingleLayerRegressor(config)
    elif model_name == "multi_layer_regressor":
        model = MultiLayerRegressor(config)
    elif is_transformer_model(model_name):
        model = create_transformer_model(model_name, config)
    else:
        raise RuntimeError(f"Unknown model name {model_name}.")
    return model


def create_transformer_model(model_name: str, config: Optional[BaseModelConfig] = None) -> nn.Module:
    """
    Create a transformer model (e.g. BERT) either using the
    default pretrained model or using the provided config.
    """
    # Make sure model is supported
    assert is_transformer_model(model_name)

    # Import here because it's an optional dependency
    from transformers import AutoConfig, AutoModel

    if config is not None and hasattr(config, "output_dir"):  # Load trained model from config
        kwargs = {
            "pretrained_model_name_or_path": get_file_path(config.output_dir, config.model_name_or_path),
            "from_tf": False,
            "config": AutoConfig.from_pretrained(get_file_path(config.output_dir, config.model_config_path)),
        }
        model = AutoModel.from_pretrained(**kwargs)

    else:  # Load default pre-trained model
        model = AutoModel.from_pretrained(model_name)

    return model


def is_transformer_model(model_name: str) -> bool:
    """
    Check if given `model_name` is a transformer
    model by attempting to load the model config.
    Returns False if `transformers` is not installed.
    """
    try:
        # Import here because it's an optional dependency
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
        return True
    except OSError:
        return False
