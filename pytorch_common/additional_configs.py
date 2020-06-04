from munch import Munch
from copy import deepcopy


class BaseDatasetConfig(Munch):
    """
    Base configuration class for
    dataset-related settings.

    Class attributes can be accessed with both
    `configobj["key"]` and `configobj.key`.
    """
    def __init__(self, dictionary=None):
        if dictionary:
            super().__init__(dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

    def _initialize_additional_config(self):
        pass

    def copy(self, deep=True):
        """
        Return a BaseDatasetConfig object with
        same attribute values.
        Useful if different configs are
        required, e.g. for train/val datasets.
        """
        if deep:
            return deepcopy(self)
        return BaseDatasetConfig(self)


class BaseModelConfig(Munch):
    """
    Base configuration class for
    model-related settings.

    Class attributes can be accessed with both
    `configobj["key"]` and `configobj.key`.
    """

    def __init__(self, dictionary=None, model_type="classification"):
        if dictionary:
            super().__init__(dictionary)

            # Set Config values that are not in config yaml
            self._initialize_additional_config()

        self.model_type = model_type

    def _initialize_additional_config(self):
        pass

    def copy(self, deep=True):
        """
        Return a BaseModelConfig object with
        same attribute values.
        Useful if different configs are required.
        """
        if deep:
            return deepcopy(self)
        return BaseModelConfig(self)
