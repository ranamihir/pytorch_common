import logging
import os
import torch
from munch import Munch

import pytorch_common
from pytorch_common import metrics
from .utils import load_object, make_dirs, set_seed


class Config(Munch):
    '''
    Configuration class that can be used to have fields for the
    configuration instead of just going off the dictionary.

    This class extends dict so values can be accessed
    with both `configobj['key']` and `configobj.key`.
    '''
    def __init__(self, dictionary=None):
        if dictionary:
            super().__init__(dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

    def _initialize_additional_config(self):
        pass


def load_pytorch_common_config(dictionary):
    '''
    Loads the pytorch_common config (if present) and
    updates attributes which are present in the project
    specific `dictionary`, and returns that dictionary.
    '''
    # Load pytorch_common config if required
    if dictionary.get('load_pytorch_common_config'):
        logging.info('Loading default pytorch_common config...')
        pytorch_common_config = load_config()

        logging.info('Done. Overriding default config with provided dictionary...')

        # Override pytorch_common config with project specific one
        # And then set it back to original dictionary
        pytorch_common_config.update(dictionary)
        merged_config = pytorch_common_config
        logging.info('Done.')

        # Throw warning if both scheduler configs enabled (not common)
        if merged_config.use_scheduler_after_step and merged_config.use_scheduler_after_epoch:
            logging.warning('Scheduler is configured to take a step '
                            'both after every step and every epoch.')

        # Throw warning if checkpointing is disabled
        if merged_config.disable_checkpointing:
            logging.warning('Checkpointing is disabled. No models will be saved during training.')
    return merged_config

def load_config(config_file='config.yaml'):
    '''
    Load pytorch_common config.
    Used in other repositories for loading
    this config alongside their own configs
    to avoid repeating attributes.
    '''
    # Create and initialize the runner
    packagedir = pytorch_common.__path__[0]
    configdir = os.path.join(packagedir, 'configs')

    # Load pytorch_common config
    dictionary = load_object(configdir, config_file, module='yaml')
    config = Config(dictionary)
    return config

def set_pytorch_config(config):
    '''
    Validate and set config for all
    things related to PyTorch / GPUs.
    '''
    if config.get('load_pytorch_common_config'):
        set_additional_dirs(config) # Set and create additional required directories

        # Set and validate loss and eval criteria
        set_loss_and_eval_criteria(config)

        # Verify config for CUDA / CPU device(s) provided
        check_and_set_devices(config)

        # Compute correct batch size if per GPU one available
        set_batch_size(config)

        # Fix seed
        set_seed(config.seed)

        # Check for model and classification type
        assert (config.model_type == 'classification' and \
            config.classification_type in ['binary', 'multiclass', 'multilabel']) \
            or (config.model_type == 'regression' and not hasattr(config, 'classification_type'))

        # TODO: Remove this after extending FocalLoss
        if config.model_type == 'classification' and config.loss_criterion == 'focal-loss':
            assert config.classification_type == 'binary'

        # Ensure GPU availability as some models are prohibitively slow on CPU
        if config.assert_gpu:
            assert config.n_gpu >= 1, ("Usage of GPU is required as per config but either "
                                       "one isn't available or the device is set to CPU.")

def set_additional_dirs(config):
    '''
    Update `output_dir`, `plot_dir`, and `checkpoint_dir`
    directory paths to absolute ones and create them.
    '''
    if config.get('misc_data_dir'):
        set_and_create_dir(config, config.packagedir, 'misc_data_dir')
    else: # Point misc_data_dir to transientdir by default
        config['misc_data_dir'] = config.transientdir
        setattr(config, 'misc_data_dir', config.transientdir)

    for directory in ['output_dir', 'plot_dir', 'checkpoint_dir']:
        if config.get(directory):
            set_and_create_dir(config, config.misc_data_dir, directory)

def set_and_create_dir(config, parent_dir, directory):
    '''
    Properly sets the `directory` attribute of `config`,
    assuming `config[directory]` is inside `parent_dir`.
    And creates the directory if it doesn't already exist.
    '''
    dir_path = os.path.expanduser(os.path.join(parent_dir, config[directory]))
    config[directory] = dir_path
    setattr(config, directory, dir_path)
    make_dirs(config[directory])

def set_loss_and_eval_criteria(config):
    '''
    Create loss and evaluation criteria
    as per their (optionally) provided
    respective kwargs.
    '''
    # Set loss and eval criteria kwargs
    config.loss_kwargs = config.loss_kwargs if config.get('loss_kwargs') else {}
    config.eval_criteria_kwargs = config.eval_criteria_kwargs \
                                  if config.get('eval_criteria_kwargs') else {}

    # Check for evaluation criteria
    assert config.get('eval_criteria') and isinstance(config.eval_criteria, list)
    for eval_criterion in config.eval_criteria:
        assert eval_criterion in metrics.EVAL_CRITERIA

    # If early stopping not used, the criterion is still
    # defined just for getting the "best" epoch
    if config.use_early_stopping:
        assert config.early_stopping_criterion is not None
    else:
        default_stopping_criterion = 'mse' if config.model_type == 'regression' else 'accuracy'
        config.early_stopping_criterion = default_stopping_criterion
    assert config.early_stopping_criterion in config.eval_criteria

def check_and_set_devices(config):
    '''
    Check the validity of provided device
    configuration:
      - Properly set fields like `device`,
        `device_ids`, and `n_gpu`.
      - Set torch backend benchmarks
    '''
    # Check device provided
    if 'cuda' in config.device:
        assert torch.cuda.is_available() # Check for CUDA
        torch.cuda.set_device(config.device) # Set default CUDA device

        # Get device IDs
        config.device_ids = config.device_ids if config.get('device_ids') else []

        # Parallelize across all available GPUs if required
        if config.device_ids == -1:
            config.device_ids = list(range(torch.cuda.device_count()))

        # Set number of GPUs to be used
        config.n_gpu = len(config.device_ids) if len(config.device_ids) else 1
        assert config.n_gpu <= torch.cuda.device_count()
    else:
        assert config.device == 'cpu'
        config.device_ids = []
        config.n_gpu = 0

    # Make sure default device is consistent if parallelized
    if config.n_gpu > 1:
        # Get default device index
        try:
            default_device = int(config.device.split(':')[1])
        except:
            default_device = 0

        # Swap order if necessary to bring default_device to index 0
        default_device_ind = config.device_ids.index(default_device)
        config.device_ids[default_device_ind], config.device_ids[0] = \
            config.device_ids[0], config.device_ids[default_device_ind]

    # Use cudnn benchmarks
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def set_batch_size(config):
    '''
    If batch size per GPU is provided and the model
    is to be parallelized, then compute the corresponding
    total batch size.
    '''
    if config.get('batch_size_per_gpu'):
        if config.get('batch_size'):
            raise ValueError(f"Please don't provide both \"batch_size\" and "
                             f"\"batch_size_per_gpu\" at the same time.")

        # Set correct batch size according to number of devices
        config.batch_size = config.batch_size_per_gpu # if CPU or only 1 GPU
        if config.n_gpu > 1:
            config.batch_size *= config.n_gpu
