from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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
    "_Config",
    "_Device",
    "_Batch",
    "_Loss",
    "_Figure",
    "_TensorOrTensors",
    "_ModelOrModels",
    "_EvalCriterionOrCriteria",
    "_TrainResult",
    "_EvalResult",
    "_TestResult",
    "_DecoupleFnTrain",
    "_DecoupleFnTest",
    "_DecoupleFn",
]


_StringDict = Dict[str, Any]
_Config = Union[_StringDict, Munch]
_Device = Union[str, torch.device]
_Batch = Iterable
_Figure = Figure

_TensorOrTensors = Union[torch.Tensor, Iterable[torch.Tensor]]
_ModelOrModels = Union[nn.Module, Iterable[nn.Module]]
_EvalCriterionOrCriteria = Union[Dict[str, Callable], Dict[str, Iterable[Callable]]]

_TrainResult = List[float]
_EvalResult = Tuple[List[float], Dict[str, float], torch.Tensor, torch.Tensor]
_TestResult = Iterable[torch.Tensor]

_DecoupleFnTrain = Callable[[_Batch], Tuple[_Batch]]
_DecoupleFnTest = Callable[[_Batch], _Batch]
_DecoupleFn = Callable[[_Batch], Union[_Batch, Tuple[_Batch]]]
