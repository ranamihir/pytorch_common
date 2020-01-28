"""
Sample config.py for loading configuration from yaml files
"""
from vrdscommon.dsprunner import DspRunner
from vrdscommon import common_utils
from datetime import datetime, timedelta
import os
import torch
import pkg_resources
import yaml

from .additional_configs import BaseDatasetConfig, BaseModelConfig
from .utils import make_dirs, set_seed


class Config(common_utils.CommonConfiguration):
    """
    Configuration class that can be used to have fields for the
    configuration instead of just going off the dictionary.

    This class extends dict so values can be accessed in the same manner as a dictionary, like configobj['key'].

    Common variables on the superclass to be accessed:

    >>> configobj.datadir
    >>> configobj.exportdir
    >>> configobj.artifactdir
    >>> configobj.transformdir
    """

    def __init__(self, dictionary=None):
        if dictionary:
            common_utils.CommonConfiguration.__init__(self, dictionary)

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
        pytorch_common_config = load_config()

        # Override pytorch_common config with project specific one
        # And then set it back to original dictionary
        pytorch_common_config.update(dictionary)
        dictionary = pytorch_common_config
    return dictionary

def load_config(config_file='config.yaml'):
    '''
    Load pytorch_common config.
    Used in other repositories for loading
    this config alongside their own configs
    to avoid repeating attributes.
    '''
    # Create and initialize the runner
    runner = DspRunner('pytorch_common')
    runner.set_config_class(Config)
    packagedir = os.path.normpath(pkg_resources.resource_filename('pytorch_common', '.'))
    configdir = os.path.join(packagedir, 'configs')

    # Load pytorch_common config
    config_file_path = os.path.join(configdir, config_file)
    if os.path.exists(config_file_path):
        with open(config_file_path, 'r') as f:
            config_dict = yaml.load(f)
        return config_dict
    return {}

def set_pytorch_config(config):
    set_additional_dirs(config) # Set and create additional required directories

    # Set and validate loss and eval criteria
    set_loss_and_eval_criteria(config)

    # Verify config for CUDA / CPU device(s) provided
    check_and_set_devices(config)

    # Fix seed
    set_seed(config)

    # Check for model type
    if config.model_type not in ['classification', 'regression']:
        raise ValueError(f'Param "model_type" ("{config.model_type}") must be '
                         f'one of ["classification", "regression"]')

    # Check for classification type
    if config.model_type == 'classification':
        assert config.classification_type in ['binary', 'multiclass', 'multilabel']

        # TODO: Remove this after extending FocalLoss
        if config.loss_criterion == 'focal-loss':
            assert config.classification_type == 'binary'

def set_additional_dirs(obj):
    # Update directory paths to absolute ones and create them
    for directory in ['output_dir', 'plot_dir', 'checkpoint_dir']:
        if hasattr(obj, directory):
            dir_path = os.path.expanduser(os.path.join(obj.transientdir, getattr(obj, directory)))
            obj[directory] = dir_path
            setattr(obj, directory, dir_path)
            make_dirs(obj[directory])

def set_loss_and_eval_criteria(obj):
    # Set loss and eval criteria kwargs
    obj.loss_kwargs = obj.loss_kwargs if obj.get('loss_kwargs') else {}
    obj.eval_criteria_kwargs = obj.eval_criteria_kwargs if obj.get('eval_criteria_kwargs') else {}

    # Check for evaluation criteria
    allowed_eval_metrics = ['mse', 'accuracy', 'precision', 'recall', 'f1', 'auc']
    assert hasattr(obj, 'eval_criteria') and isinstance(obj.eval_criteria, list)
    for eval_criterion in obj.eval_criteria:
        assert eval_criterion in allowed_eval_metrics
    default_stopping_criterion = 'accuracy' if obj.model_type == 'classification' else 'mse'
    obj.early_stopping_criterion = obj.get('early_stopping_criterion', default_stopping_criterion)
    assert obj.early_stopping_criterion in obj.eval_criteria

def check_and_set_devices(obj):
    # Check device provided
    if 'cuda' in obj.device:
        assert torch.cuda.is_available() # Check for CUDA
        torch.cuda.set_device(obj.device) # Set default CUDA device

        # Get device IDs
        obj.device_ids = obj.device_ids if obj.get('device_ids') else []

        # Parallelize across all available GPUs
        if obj.device_ids == -1:
            obj.device_ids = list(range(torch.cuda.device_count()))

        obj.n_gpu = len(obj.device_ids) if len(obj.device_ids) else 1 # Set number of GPUs to be used
        assert obj.n_gpu <= torch.cuda.device_count()
    else:
        assert obj.device == 'cpu'
        obj.device_ids = []
        obj.n_gpu = 0

    # Make sure default device is consistent if parallelized
    if obj.n_gpu > 1:
        # Get default device index
        try:
            default_device = int(obj.device.split(':')[1])
        except:
            default_device = 0

        # Swap order if necessary to bring default_device to index 0
        default_device_ind = obj.device_ids.index(default_device)
        obj.device_ids[default_device_ind], obj.device_ids[0] = obj.device_ids[0], obj.device_ids[default_device_ind]

    # Use cudnn benchmarks
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
