from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from munch import Munch
from torch.nn.modules.loss import _Loss

__all__ = [
    "Any",
    "List",
    "Tuple",
    "Dict",
    "Iterable",
    "Callable",
    "Optional",
    "Union",
    "Munch",
    "_StringDict",
    "_StringArrayDict",
    "_Config",
    "_Device",
    "_Batch",
    "_Loss",
    "Figure",
    "_TensorOrArray",
    "_TensorsOrArrays",
    "_TensorOrTensors",
    "_ModelOrModels",
    "_EvalCriterionOrCriteria",
    "_DecoupleFnTrain",
    "_DecoupleFnTest",
    "_DecoupleFn",
]


_StringDict = Dict[str, Any]
_StringArrayDict = Dict[str, np.ndarray]
_Config = Union[_StringDict, Munch]
_Device = Union[str, torch.device]
_Batch = Iterable

_TensorOrArray = Union[torch.Tensor, np.ndarray]
_TensorsOrArrays = Union[Iterable[torch.Tensor], Iterable[np.ndarray]]
_TensorOrTensors = Union[torch.Tensor, Iterable[torch.Tensor]]
_ModelOrModels = Union[nn.Module, Iterable[nn.Module]]
_EvalCriterionOrCriteria = Union[Dict[str, Callable], Dict[str, Iterable[Callable]]]

_DecoupleFnTrain = Callable[[_Batch], Tuple[_Batch]]
_DecoupleFnTest = Callable[[_Batch], _Batch]
_DecoupleFn = Callable[[_Batch], Union[_Batch, Tuple[_Batch]]]
