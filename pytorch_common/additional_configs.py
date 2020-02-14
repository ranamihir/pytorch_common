"""
Sample config.py for loading configuration from yaml files
"""
from vrdscommon import common_utils
from copy import deepcopy


class BaseDatasetConfig(common_utils.CommonConfiguration):
    '''
    Base Dataset Configuration class that can be used to have fields
    for the configuration instead of just going off the dictionary.

    This class extends dict so values can be accessed in the same manner
    as a dictionary, like configobj['key'].
    '''

    def __init__(self, dictionary=None):
        if dictionary:
            common_utils.CommonConfiguration.__init__(self, dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

    def _initialize_additional_config(self):
        pass

    def copy(self, deep=True):
        '''
        Return a BaseDatasetConfig object with
        same attribute values.
        Useful if different configs are
        required, e.g. for train/val datasets.
        '''
        if deep:
            return deepcopy(self)
        return BaseDatasetConfig(self)


class BaseModelConfig(common_utils.CommonConfiguration):
    '''
    Base Model Configuration class that can be used to have fields
    for the configuration instead of just going off the dictionary.

    This class extends dict so values can be accessed in the same manner
    as a dictionary, like configobj['key'].
    '''

    def __init__(self, dictionary=None, model_type='classification'):
        if dictionary:
            common_utils.CommonConfiguration.__init__(self, dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

        self.model_type = model_type

    def _initialize_additional_config(self):
        pass

    def copy(self, deep=True):
        '''
        Return a BaseModelConfig object with
        same attribute values.
        Useful if different configs are required.
        '''
        if deep:
            return deepcopy(self)
        return BaseModelConfig(self)
