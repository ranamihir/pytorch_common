from munch import Munch
from matplotlib.figure import Figure
from typing import Any, List, Tuple, Dict, Callable, Iterable, Optional, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


_string_dict = Dict[str, Any]
_config = Union[_string_dict, Munch]
_device = Union[str, torch.device]
_batch = Iterable
_figure = Figure

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
_model_or_models = Union[nn.Module, List[nn.Module]]
_loss_or_losses = Union[_Loss, Iterable[_Loss]]
_eval_criterion_or_criteria = Union[Dict[str, Callable], Dict[str, List[Callable]]]

_train_result = List[float]
_eval_result = Tuple[List[float], Dict[str, float], torch.Tensor, torch.Tensor]
_test_result = Iterable[torch.Tensor]
