from __future__ import annotations

import os

import torch
from munch import Munch

import pytorch_common

from .metric_utils import CLASSIFICATION_LOSS_CRITERIA, REGRESSION_LOSS_CRITERIA
from .metrics import EvalCriteria
from .types import Optional, _Config, _StringDict
from .utils import get_file_path, load_object, make_dirs, set_seed, setup_logging

logger = setup_logging(__name__)


class Config(Munch):
    """
    Configuration class for PyTorch-related settings.

    Class attributes can be accessed with both
    `configobj["key"]` and `configobj.key`.
    """

    def __init__(self, dictionary: Optional[_StringDict] = None) -> None:
        if dictionary:
            super().__init__(dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

    def _initialize_additional_config(self) -> None:
        pass


def load_pytorch_common_config(dictionary: Optional[_StringDict] = None) -> Munch:
    """
    Load the pytorch_common config (if present) and
    update attributes which are present in the project
    specific `dictionary`, and return that dictionary.
    """
    if dictionary is None:
        dictionary = {}

    # Load `pytorch_common` config if required
    logger.info("Loading default pytorch_common config...")
    pytorch_common_config = load_config()

    logger.info("Done. Overriding default config with provided dictionary...")

    # Override pytorch_common config with project specific one
    # And then set it back to original dictionary
    pytorch_common_config.update(dictionary)
    merged_config = pytorch_common_config
    logger.info("Done.")

    # Throw warning if both scheduler configs enabled (not common)
    if merged_config.use_scheduler_after_step and merged_config.use_scheduler_after_epoch:
        logger.warning("Scheduler is configured to take a step both after every step and every epoch.")

    # Throw warning if checkpointing is disabled
    if merged_config.disable_checkpointing:
        logger.warning("Checkpointing is disabled. No models will be saved during training.")

    set_pytorch_config(merged_config)

    return merged_config


def load_config(config_file: Optional[str] = "config.yaml") -> Config:
    """
    Load pytorch_common config.
    Used in other repositories for loading
    this config alongside their own configs
    to avoid repeating attributes.
    """
    # Create and initialize the runner
    package_dir = pytorch_common.__path__[0]
    config_dir = get_file_path(package_dir, "configs")

    # Load pytorch_common config
    dictionary = load_object(config_dir, config_file, module="yaml")
    config = Config(dictionary)
    return config


def set_pytorch_config(config: _Config) -> None:
    """
    Validate and set config for all
    things related to PyTorch / GPUs.
    """
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

    # Check miscellaneous configurations
    check_and_set_misc_config(config)

    # Ensure GPU availability as some models are prohibitively slow on CPU
    if config.assert_gpu:
        assert (
            config.n_gpu >= 1
        ), "Usage of GPU is required as per config but either one isn't available or the device is set to CPU."


def set_additional_dirs(config: _Config) -> None:
    """
    Update all directory paths to
    absolute ones and create them.
    """
    if not config.disable_checkpointing:
        set_and_create_dir(config, "artifact_dir")
        for directory in ["output_dir", "plot_dir", "checkpoint_dir", "log_dir"]:
            if config.get(directory):
                set_and_create_dir(config, directory, config.artifact_dir)


def set_and_create_dir(config: _Config, directory: str, parent_dir: Optional[str] = None) -> None:
    """
    Properly sets the `directory` attribute of `config`,
    assuming `config[directory]` is inside `parent_dir` (if provided).
    Also creates the directory if it doesn't already exist.
    """
    dir_path = config[directory] if parent_dir is None else get_file_path(parent_dir, config[directory])
    dir_path = os.path.expanduser(dir_path)
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
            primary_criterion = "mse" if config.model_type == "regression" else "accuracy"
            config.early_stopping_criterion = primary_criterion
    assert config.early_stopping_criterion in config.eval_criteria


def _check_loss_and_eval_criteria(config: _Config) -> None:
    """
    Ensure that the loss and eval criteria
    provided are consistent with the
    specified `model_type`.
    """
    assert config.get("eval_criteria") and isinstance(config.eval_criteria, list)

    loss_criteria = CLASSIFICATION_LOSS_CRITERIA if config.model_type == "classification" else REGRESSION_LOSS_CRITERIA
    assert config.loss_criterion in loss_criteria, (
        f"Loss criterion ('{config.loss_criterion}') "
        f"for `model_type=='classification' must be one"
        f" of {loss_criteria}."
    )

    supported_criteria = EvalCriteria(model_type=config.model_type)
    for eval_criterion in config.eval_criteria:
        assert eval_criterion in supported_criteria, (
            f"Eval criterion ('{eval_criterion}') for "
            f"`model_type=='{config.model_type}'` must be one"
            f" of {supported_criteria.names}."
        )


def check_and_set_devices(config: _Config) -> None:
    """
    Check the validity of provided device configuration:
      - Properly set fields like `device`,
        `device_ids`, and `n_gpu`.
      - Set torch backend benchmarks
    """
    # Check device provided
    if "cuda" in config.device:
        assert torch.cuda.is_available()  # Check for CUDA
        torch.cuda.set_device(config.device)  # Set default CUDA device

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
        assert not config.get("device_ids") or config.device_ids == -1
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
        config.device_ids[default_device_ind], config.device_ids[0] = (
            config.device_ids[0],
            config.device_ids[default_device_ind],
        )

    # Use cudnn benchmarks
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def set_all_batch_sizes(config: _Config) -> None:
    """
    Properly set train / eval / test batch sizes.
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
                raise ValueError(f"One of 'batch_size_per_gpu'  or '{mode_batch_size_str}' must be provided.")
            batch_size_to_set = mode_batch_size_per_gpu  # Specific to each mode
        else:
            if mode_batch_size_per_gpu is not None:
                raise ValueError(
                    f"Only one of 'batch_size_per_gpu' ({batch_size_per_gpu}) or "
                    f"'{mode_batch_size_str}' ({mode_batch_size_per_gpu}) "
                    f"must be provided."
                )
            batch_size_to_set = batch_size_per_gpu  # Common for all modes

        # Set per-GPU and total batch size for mode
        set_mode_batch_size(mode, batch_size_to_set)


def check_and_set_misc_config(config: _Config) -> None:
    """
    Check all miscellaneous configurations, e.g.:
      - model_type
      - classification_type
    """
    # Check for model and classification type
    assert (
        config.model_type == "classification" and config.classification_type in ["binary", "multiclass", "multilabel"]
    ) or (config.model_type == "regression")

    # Set classification_type to None if regression
    if config.model_type == "regression":
        config.classification_type = None

    # TODO: Remove this after extending FocalLoss
    if config.model_type == "classification" and config.loss_criterion == "focal-loss":
        assert (
            config.classification_type == "binary"
        ), "FocalLoss is currently only supported for binary classification."

    # Used for dataloader sampling
    config.num_batches = config.get("num_batches", None)
    config.percentage = config.get("percentage", None)
