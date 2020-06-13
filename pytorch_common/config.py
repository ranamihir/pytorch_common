from __future__ import annotations
import logging
import os
import torch
from munch import Munch

import pytorch_common
from .metrics import (
    CLASSIFICATION_LOSS_CRITERIA, CLASSIFICATION_EVAL_CRITERIA,
    REGRESSION_LOSS_CRITERIA, REGRESSION_EVAL_CRITERIA
)
from .utils import get_file_path, make_dirs, set_seed, load_object
from .types import Optional, _StringDict, _Config


class Config(Munch):
    """
    Configuration class for PyTorch-related settings.

    Class attributes can be accessed with both
    `configobj["key"]` and `configobj.key`.
    """
    def __init__(self, dictionary: Optional[_StringDict] = None):
        if dictionary:
            super().__init__(dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

    def _initialize_additional_config(self):
        pass


def load_pytorch_common_config(dictionary: _StringDict) -> Munch:
    """
    Load the pytorch_common config (if present) and
    update attributes which are present in the project
    specific `dictionary`, and return that dictionary.
    """
    # Load `pytorch_common` config if required
    if dictionary.get("load_pytorch_common_config"):
        logging.info("Loading default pytorch_common config...")
        pytorch_common_config = load_config()

        logging.info("Done. Overriding default config with provided dictionary...")

        # Override pytorch_common config with project specific one
        # And then set it back to original dictionary
        pytorch_common_config.update(dictionary)
        merged_config = pytorch_common_config
        logging.info("Done.")

        # Throw warning if both scheduler configs enabled (not common)
        if merged_config.use_scheduler_after_step and merged_config.use_scheduler_after_epoch:
            logging.warning(
                "Scheduler is configured to take a step both after every step and every epoch."
            )

        # Throw warning if checkpointing is disabled
        if merged_config.disable_checkpointing:
            logging.warning("Checkpointing is disabled. No models will be saved during training.")

        return Munch(merged_config)
    return Munch(dictionary)

def load_config(config_file: Optional[str] = "config.yaml") -> Config:
    """
    Load pytorch_common config.
    Used in other repositories for loading
    this config alongside their own configs
    to avoid repeating attributes.
    """
    # Create and initialize the runner
    packagedir = pytorch_common.__path__[0]
    configdir = get_file_path(packagedir, "configs")

    # Load pytorch_common config
    dictionary = load_object(configdir, config_file, module="yaml")
    config = Config(dictionary)
    return config

def set_pytorch_config(config: _Config) -> None:
    """
    Validate and set config for all
    things related to PyTorch / GPUs.
    """
    if config.get("load_pytorch_common_config"):
        # Set and create additional required directories
        set_additional_dirs(config)

        # Set and validate loss and eval criteria
        set_loss_and_eval_criteria(config)

        # Verify config for CUDA / CPU device(s) provided
        check_and_set_devices(config)

        # Compute correct batch size if per GPU one available
        set_all_batch_sizes(config)

        # Fix seed
        set_seed(config.seed)

        # Check for model and classification type
        assert (config.model_type == "classification" and \
            config.classification_type in ["binary", "multiclass", "multilabel"]) \
            or (config.model_type == "regression" and not hasattr(config, "classification_type"))

        # TODO: Remove this after extending FocalLoss
        if config.model_type == "classification" and config.loss_criterion == "focal-loss":
            assert config.classification_type == "binary", ("FocalLoss is currently only supported"
                                                            "for binary classification.")

        # Ensure GPU availability as some models are prohibitively slow on CPU
        if config.assert_gpu:
            assert config.n_gpu >= 1, ("Usage of GPU is required as per config but either "
                                       "one isn't available or the device is set to CPU.")

def set_additional_dirs(config: _Config) -> None:
    """
    Update `output_dir`, `plot_dir`, and `checkpoint_dir`
    directory paths to absolute ones and create them.
    """
    if config.get("misc_data_dir"):
        set_and_create_dir(config, config.packagedir, "misc_data_dir")
    else: # Point `misc_data_dir` to `transientdir` by default
        config["misc_data_dir"] = config.transientdir
        setattr(config, "misc_data_dir", config.transientdir)

    for directory in ["output_dir", "plot_dir", "checkpoint_dir"]:
        if config.get(directory):
            set_and_create_dir(config, config.misc_data_dir, directory)

def set_and_create_dir(config: _Config, parent_dir: str, directory: str) -> None:
    """
    Properly sets the `directory` attribute of `config`,
    assuming `config[directory]` is inside `parent_dir`.
    Also creates the directory if it doesn't already exist.
    """
    dir_path = os.path.expanduser(get_file_path(parent_dir, config[directory]))
    config[directory] = dir_path
    setattr(config, directory, dir_path)
    make_dirs(config[directory])

def set_loss_and_eval_criteria(config: _Config) -> None:
    """
    Create loss and evaluation criteria
    as per their (optionally) provided
    respective kwargs.
    """
    # Set loss and eval criteria kwargs
    # This logic allows leaving their values empty even if their keys are specified
    if not config.get("loss_kwargs"):
        config.loss_kwargs = {}
    if not config.get("eval_criteria_kwargs"):
        config.eval_criteria_kwargs = {}

    # Check for evaluation criteria
    _check_loss_and_eval_criteria(config)

    # If early stopping not used, the criterion is
    # still defined just for getting the "best" epoch
    if config.use_early_stopping:
        assert config.early_stopping_criterion is not None
    else:
        if not hasattr(config, "early_stopping_criterion"):
            default_stopping_criterion = "mse" if config.model_type == "regression" else "accuracy"
            config.early_stopping_criterion = default_stopping_criterion
    assert config.early_stopping_criterion in config.eval_criteria

def _check_loss_and_eval_criteria(config: _Config) -> None:
    """
    Ensure that the loss and eval criteria
    provided are consistent with the
    specified `model_type`.
    """
    assert config.get("eval_criteria") and isinstance(config.eval_criteria, list)

    if config.model_type == "classification":
        LOSS_CRITERIA = CLASSIFICATION_LOSS_CRITERIA
        EVAL_CRITERIA = CLASSIFICATION_EVAL_CRITERIA
    else:
        LOSS_CRITERIA = REGRESSION_LOSS_CRITERIA
        EVAL_CRITERIA = REGRESSION_EVAL_CRITERIA

    assert config.loss_criterion in LOSS_CRITERIA, (f"Loss criterion ('{config.loss_criterion}') "
                                                    f"for `model_type=='classification' must be one"
                                                    f" of {LOSS_CRITERIA}.")
    for eval_criterion in config.eval_criteria:
        assert eval_criterion in EVAL_CRITERIA, (f"Eval criterion ('{eval_criterion}') for "
                                                 f"`model_type=='classification'` must be one"
                                                 f" of {EVAL_CRITERIA}.")

def check_and_set_devices(config: _Config) -> None:
    """
    Check the validity of provided device configuration:
      - Properly set fields like `device`,
        `device_ids`, and `n_gpu`.
      - Set torch backend benchmarks
    """
    # Check device provided
    if "cuda" in config.device:
        assert torch.cuda.is_available() # Check for CUDA
        torch.cuda.set_device(config.device) # Set default CUDA device

        # Get device IDs
        config.device_ids = config.device_ids if config.get("device_ids") else []

        # Parallelize across all available GPUs if required
        if config.device_ids == -1:
            config.device_ids = list(range(torch.cuda.device_count()))

        # Set number of GPUs to be used
        config.n_gpu = len(config.device_ids) if len(config.device_ids) else 1
        assert config.n_gpu <= torch.cuda.device_count()
    else:
        assert config.device == "cpu"
        assert (not config.get("device_ids") or config.device_ids == -1)
        config.device_ids = []
        config.n_gpu = 0

    # Make sure default device is consistent if parallelized
    if config.n_gpu > 1:
        # Get default device index
        try:
            default_device = int(config.device.split(":")[1])
        except:
            default_device = 0

        # Swap order if necessary to bring default_device to index 0
        default_device_ind = config.device_ids.index(default_device)
        config.device_ids[default_device_ind], config.device_ids[0] = \
            config.device_ids[0], config.device_ids[default_device_ind]

    # Use cudnn benchmarks
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def set_all_batch_sizes(config: _Config) -> None:
    """
    Properly set train/eval/test batch sizes.
    The setting should be such that:
      - either only `batch_size_per_gpu` is provided,
        in which case the same batch size will be
        propagated to all dataloaders
      - or separate values may be provided for each of
        `train_batch_size_per_gpu`, `eval_batch_size_per_gpu`,
        and `test_batch_size_per_gpu`
    The per-GPU batch sizes for each mode will then be converted
    to the total batch size depending on number of devices.
    """
    def set_mode_batch_size(mode: str, batch_size_per_gpu: int) -> None:
        """
        Set per-GPU and total batch size for a
        given `mode` based on the `batch_size_per_gpu`
        and number of GPUs available.
        """
        setattr(config, f"{mode}_batch_size_per_gpu", batch_size_per_gpu)

        # Set correct batch size according to number of devices
        batch_size = max(1, config.n_gpu) * batch_size_per_gpu
        setattr(config, f"{mode}_batch_size", batch_size)

    batch_size = config.get("batch_size")
    if batch_size is not None:
        raise ValueError(
            f"Param 'batch_size' ({batch_size}) is now deprecated. Please either provide "
             "'batch_size_per_gpu' for per-GPU batch size, or 'train_batch_size_per_gpu', "
             "'eval_batch_size_per_gpu', and 'test_batch_size_per_gpu' if different per-GPU "
             "batch sizes are to be provided for each mode."
        )

    SUPPORTED_MODES = ["train", "eval", "test"]
    batch_size_per_gpu = config.get("batch_size_per_gpu")

    for mode in SUPPORTED_MODES:
        mode_batch_size_str = f"{mode}_batch_size_per_gpu"
        mode_batch_size_per_gpu = config.get(mode_batch_size_str)
        if batch_size_per_gpu is None:
            if mode_batch_size_per_gpu is None:
                raise ValueError(
                    f"One of 'batch_size_per_gpu'  or '{mode_batch_size_str}' must be provided."
                )
            batch_size_to_set = mode_batch_size_per_gpu # Specific to each mode
        else:
            if mode_batch_size_per_gpu is not None:
                raise ValueError(
                    f"Only one of 'batch_size_per_gpu' ({batch_size_per_gpu}) or "
                    f"'{mode_batch_size_str}' ({mode_batch_size_per_gpu}) "
                    f"must be provided."
                )
            batch_size_to_set = batch_size_per_gpu # Common for all modes

        # Set per-GPU and total batch size for mode
        set_mode_batch_size(mode, batch_size_to_set)
