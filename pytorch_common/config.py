"""
Sample config.py for loading configuration from yaml files
"""
from vrdscommon import common_utils
from datetime import datetime, timedelta
import os
import torch
import pkg_resources

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

        def is_prod():
            return common_utils.vrdsprefix() == 'prod'

        # Set package dir and config dir
        self.packagedir = os.path.normpath(pkg_resources.resource_filename(self.package, '.'))
        self.configdir = os.path.join(self.packagedir, 'configs')

        # Set dataset and model configs
        self.dataset_config = BaseDatasetConfig(self.dataset_config)
        self.model_config = BaseModelConfig(self.model_config, self.model_type)

        self._set_additional_dirs() # Set and create additional required directories

        # Set and validate loss and eval criteria
        self._set_loss_and_eval_criteria()

        # Verify config for CUDA / CPU device(s) provided
        self._check_and_set_devices()

        # Fix seed
        set_seed(self)

        # Check for model type
        if self.model_type not in ['classification', 'regression']:
            raise ValueError('Param "model_type" ("{}") must be one of ["classification", '\
                             '"regression"]'.format(self.model_type))

        # Check for classification type
        if self.model_type == 'classification':
            assert self.classification_type in ['binary', 'multiclass', 'multilabel']

            # TODO: Remove this after extending FocalLoss
            if self.loss_criterion == 'focal-loss':
                assert self.classification_type == 'binary'

    def _set_additional_dirs(self):
        # Update directory paths to absolute ones and create them
        for directory in ['output_dir', 'plot_dir', 'checkpoint_dir']:
            if hasattr(self, directory):
                dir_path = os.path.expanduser(os.path.join(self.transientdir, self[directory]))
                self[directory] = dir_path
                setattr(self, directory, dir_path)
                make_dirs(self[directory])

    def _set_loss_and_eval_criteria(self):
        # Set loss and eval criteria kwargs
        self.loss_kwargs = self.loss_kwargs if self.get('loss_kwargs') else {}
        self.eval_criteria_kwargs = self.eval_criteria_kwargs if self.get('eval_criteria_kwargs') else {}

        # Check for evaluation criteria
        allowed_eval_metrics = ['mse', 'accuracy', 'precision', 'recall', 'f1', 'auc']
        assert hasattr(self, 'eval_criteria') and isinstance(self.eval_criteria, list)
        for eval_criterion in self.eval_criteria:
            assert eval_criterion in allowed_eval_metrics
        default_stopping_criterion = 'accuracy' if self.model_type == 'classification' else 'mse'
        self.early_stopping_criterion = self.get('early_stopping_criterion', default_stopping_criterion)
        assert self.early_stopping_criterion in self.eval_criteria

    def _check_and_set_devices(self):
        # Check device provided
        if 'cuda' in self.device:
            assert torch.cuda.is_available() # Check for CUDA
            torch.cuda.set_device(self.device) # Set default CUDA device

            # Get device IDs
            self.device_ids = self.device_ids if self.get('device_ids') else []

            # Parallelize across all available GPUs
            if self.device_ids == -1:
                self.device_ids = list(range(torch.cuda.device_count()))

            self.n_gpu = len(self.device_ids) if len(self.device_ids) else 1 # Set number of GPUs to be used
            assert self.n_gpu <= torch.cuda.device_count()
        else:
            assert self.device == 'cpu'
            self.device_ids = []
            self.n_gpu = 0

        # Make sure default device is consistent if parallelized
        if self.n_gpu > 1:
            # Get default device index
            try:
                default_device = int(self.device.split(':')[1])
            except:
                default_device = 0

            # Swap order if necessary to bring default_device to index 0
            default_device_ind = self.device_ids.index(default_device)
            self.device_ids[default_device_ind], self.device_ids[0] = self.device_ids[0], self.device_ids[default_device_ind]

        # Use cudnn benchmarks
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
