from munch import Munch
from matplotlib.figure import Figure
from typing import Any, List, Tuple, Dict, Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


_StringDict = Dict[str, Any]
_Config = Union[_StringDict, Munch]
_Device = Union[str, torch.device]
_Batch = Iterable
_Figure = Figure

_TensorOrTensors = Union[torch.Tensor, Iterable[torch.Tensor]]
_ModelOrModels = Union[nn.Module, List[nn.Module]]
_LossOrLosses = Union[_Loss, Iterable[_Loss]]
_EvalCriterionOrCriteria = Union[Dict[str, Callable], Dict[str, List[Callable]]]

_TrainResult = List[float]
_EvalResult = Tuple[List[float], Dict[str, float], torch.Tensor, torch.Tensor]
_TestResult = Iterable[torch.Tensor]

_DecoupleFnTrain = Callable[[_Batch], Tuple[_Batch]]
_DecoupleFnTest = Callable[[_Batch], _Batch]
_DecoupleFn = Callable[[_Batch], Union[_Batch, Tuple[_Batch]]]
