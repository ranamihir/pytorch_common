from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random
import shutil
import sys
from collections import OrderedDict

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from dask.callbacks import Callback
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from .types import *


def make_dirs(parent_dir_path: str, child_dirs: Optional[Union[str, List[str]]] = None) -> None:
    """
    Create the parent and (optionally) all child
    directories within parent directory.
    """

    def create_dir_if_not_exists(dir_path: str) -> None:
        """
        Create a directory at `dir_path`
        if it doesn't exist already.
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    # Create parent dir
    create_dir_if_not_exists(parent_dir_path)

    # Create child dir(s) if provided
    if child_dirs is not None:
        if isinstance(child_dirs, str):
            child_dirs = [child_dirs]
        assert isinstance(child_dirs, list)
        for dir_name in child_dirs:
            dir_path = get_file_path(parent_dir_path, dir_name)
            create_dir_if_not_exists(dir_path)


def remove_dir(dir_path: str, force: Optional[bool] = False) -> None:
    """
    Remove a directory at `dir_path`.
    :param force: whether to delete the directory
                  even if it is not empty.
                  If False and directory is not
                  empty, raises `OSError`.
    """
    if os.path.isdir(dir_path):
        if force:
            shutil.rmtree(dir_path, ignore_errors=True)
        else:
            os.rmdir(dir_path)


def human_time_interval(time_seconds: float) -> str:
    """
    Converts a time interval in seconds to a human-friendly
    representation in hours, minutes, seconds and milliseconds.
    :param time_seconds: time in seconds (float)

    >>> human_time_interval(13301.1)
    "3h 41m 41s 100ms"
    """
    hours, time_seconds = divmod(time_seconds, 3600)
    minutes, time_seconds = divmod(time_seconds, 60)
    seconds, milliseconds = divmod(time_seconds, 1)
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    milliseconds, float_milliseconds = int(milliseconds * 1000), milliseconds * 1000

    if hours > 0:
        return f"{hours}h {minutes:02}m {seconds:02}s {milliseconds:03}ms"
    if minutes > 0:
        return f"{minutes}m {seconds:02}s {milliseconds:03}ms"
    if seconds > 0:
        return f"{seconds}s {milliseconds:03}ms"
    return f"{float_milliseconds:.2f}ms"


def set_seed(seed: Optional[int] = 0) -> None:
    """
    Fix all random seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe to call even if no GPU available


def print_dataframe(data: pd.DataFrame) -> None:
    """
    Print useful summary statistics of a dataframe.
    """
    logging.info(f"\nHead of data:\n{data.head(10)}\n")
    logging.info(f"\nShape of data: {data.shape}\n")
    logging.info(f"\nColumns:\n{data.columns}\n")
    logging.info(f"\nSummary statistics:\n{data.describe()}\n")


def save_plot(
    config: _Config,
    fig: Figure,
    plot_name: str,
    model_name: str,
    config_info_dict: Optional[_StringDict] = None,
    ext: Optional[str] = "png",
) -> None:
    """
    Save a high-quality plot created by matplotlib.
    :param plot_name: Plot name, e.g. "accuracy-vs-epochs"
    :param ext: file extension
    """
    assert ext in ["png", "jpeg", "eps", "pdf"]
    unique_name = get_unique_config_name(model_name, config_info_dict)
    file_name = "-".join([plot_name, unique_name])
    fig.savefig(get_file_path(config.plot_dir, f"{file_name}.{ext}"), dpi=300)


def save_object(
    obj: Any, primary_path: str, file_name: Optional[str] = None, module: Optional[str] = "pickle"
) -> None:
    """
    This is a generic function to save any given
    object using different `module`s, e.g. pickle,
    dill, and yaml.

    Note: See `get_file_path()` for details on how
          how to set `primary_path` and `file_name`.
    """
    file_path = get_file_path(primary_path, file_name)
    logging.info(f"Saving '{file_path}'...")
    if module == "yaml":
        save_yaml(obj, file_path)
    else:
        save_pickle(obj, file_path, module)
    logging.info("Done.")


def save_pickle(obj: Any, file_path: str, module: Optional[str] = "pickle") -> None:
    """
    This is a defensive way to write (pickle/dill).dump,
    allowing for very large files on all platforms.
    """
    pickle_module = get_pickle_module(module)
    bytes_out = pickle_module.dumps(obj, protocol=pickle_module.HIGHEST_PROTOCOL)
    n_bytes = sys.getsizeof(bytes_out)
    MAX_BYTES = 2 ** 31 - 1
    with open(file_path, "wb") as f_out:
        for idx in range(0, n_bytes, MAX_BYTES):
            f_out.write(bytes_out[idx : idx + MAX_BYTES])


def save_yaml(obj: Dict, file_path: str) -> None:
    """
    Save a given dictionary as a yaml file.
    """
    assert isinstance(obj, dict), "Only `dict` objects can be stored as YAML files."
    with open(file_path, "w") as f_out:
        yaml.dump(obj, f_out)


def load_object(primary_path: str, file_name: Optional[str] = None, module: Optional[str] = "pickle") -> Any:
    """
    This is a generic function to load any given
    object using different `module`s, e.g. pickle,
    dill, and yaml.

    Note: See `get_file_path()` for details on how
          how to set `primary_path` and `file_name`.
    """
    file_path = get_file_path(primary_path, file_name)
    logging.info(f"Loading '{file_path}'...")
    if os.path.isfile(file_path):
        if module == "yaml":
            obj = load_yaml(file_path)
        else:
            obj = load_pickle(file_path, module)
        logging.info(f"Successfully loaded '{file_path}'.")
        return obj
    else:
        raise FileNotFoundError(f"Could not find '{file_path}'.")


def load_pickle(file_path: str, module: Optional[str] = "pickle") -> Any:
    """
    This is a defensive way to write (pickle/dill).load,
    allowing for very large files on all platforms.

    This function is intended to be called inside
    `load_object()`, and assumes that the file
    already exists.
    """
    input_size = os.path.getsize(file_path)
    bytes_in = bytearray(0)
    pickle_module = get_pickle_module(module)
    MAX_BYTES = 2 ** 31 - 1
    with open(file_path, "rb") as f:
        for _ in range(0, input_size, MAX_BYTES):
            bytes_in += f.read(MAX_BYTES)
    obj = pickle_module.loads(bytes_in)
    return obj


def load_yaml(file_path: str) -> Dict:
    """
    Load a given yaml file.

    Return an empty dictionary if file is empty.

    This function is intended to be called inside
    `load_object()`, and assumes that the file
    already exists.
    """
    with open(file_path, "r") as f:
        obj = yaml.safe_load(f)
    return obj if obj is not None else {}


def remove_object(primary_path: str, file_name: Optional[str] = None) -> None:
    """
    Remove a given object if it exists.

    Note: See `get_file_path()` for details on how
          how to set `primary_path` and `file_name`.
    """
    file_path = get_file_path(primary_path, file_name)
    if os.path.isfile(file_path):
        logging.info(f"Removing '{file_path}'...")
        os.remove(file_path)
        logging.info("Done.")


def get_file_path(primary_path: str, file_name: Optional[str] = None) -> str:
    """
    Generate appropriate full file path:
      - If `file_name` is None, it's assumed that the full
        path to the file is provided in `primary_path`.
      - Otherwise, it's assumed that `primary_path` is the
        path to the folder where a file named `file_name`
        exists.
    """
    return primary_path if file_name is None else os.path.join(primary_path, file_name)


def get_pickle_module(pickle_module: Optional[str] = "pickle") -> Union[pickle, dill]:
    """
    Return the correct module for pickling.
    :param pickle_module: must be one of ["pickle", "dill"]
    """
    if pickle_module == "pickle":
        return pickle
    elif pickle_module == "dill":
        return dill
    raise ValueError(f"Param 'pickle_module' ('{pickle_module}') must be one of ['pickle', 'dill'].")


def delete_model(model: nn.Module) -> None:
    """
    Delete model and free GPU memory.
    """
    model = None
    torch.cuda.empty_cache()


def get_string_from_dict(config_info_dict: Optional[_StringDict] = None) -> str:
    """
    Generate a (unique) string from a given configuration dictionary.
    The dictionary will always be sorted by key first so that if
    the order of items is changed but the dictionary is essentially
    still the same, the string returned remains unchanged.
    E.g.:
    >>> get_string_from_dict({"size": 100, "lr": 1e-3})
    "lr_0.001-size_100"
    >>> get_string_from_dict({"lr": 1e-3, "size": 100}) # Same
    "lr_0.001-size_100"
    """
    config_info = ""
    if isinstance(config_info_dict, dict):
        config_info_dict = OrderedDict(sorted(config_info_dict.items()))  # Sort to be order-agnostic
        clean = lambda k: str(k).replace("-", "_").lower()
        config_info = "-".join([f"{clean(k)}_{clean(v)}" for k, v in config_info_dict.items()])
    return config_info


def get_unique_config_name(primary_name: str, config_info_dict: Optional[_StringDict] = None) -> str:
    """
    Return a unique name for the current configuration.

    The name will comprise the `primary_name` followed by a
    hash value uniquely generated from the `config_info_dict`.
    :param primary_name: Primary name of the object being stored.
    :param config_info_dict: An optional dict provided containing
                             information about current config.

    E.g.:
    `subcategory_classifier-3d02e8616cbeab37bc1bb972ecf02882`
    Each attribute in `config_info_dict` is in the "{name}_{value}"
    format (lowercased), separated from one another by a hyphen.
    If a hyphen exists in the value (e.g. LR), it's converted to
    an underscore. Finally, this string is passed into a hash
    function to generate a unique ID for this configuration.
    """
    unique_id = ""

    # Generate unique ID based on config_info_dict
    config_info = get_string_from_dict(config_info_dict)
    if config_info != "":
        unique_id = "-" + hashlib.md5(config_info.encode("utf-8")).hexdigest()

    unique_name = primary_name + unique_id
    return unique_name


def get_checkpoint_name(
    checkpoint_type: str, model_name: str, epoch: int, config_info_dict: Optional[_StringDict] = None,
) -> str:
    """
    Returns the appropriate name of checkpoint file
    by generating a unique ID from the config.
    :param checkpoint_type: Type of checkpoint ("state" | "model")
    :param config_info_dict: An optional dict provided containing
                             information about current config.
    E.g.:
    `checkpoint-model-subcategory_classifier-3d02e8616cbeab37bc1bb972ecf02882-epoch_1.pt`
    """
    assert checkpoint_type in ["state", "model"]
    unique_name = get_unique_config_name(model_name, config_info_dict)
    checkpoint_name = f"checkpoint-{checkpoint_type}-{unique_name}-epoch_{epoch}.pt"
    return checkpoint_name


def get_trainable_params(model: nn.Module) -> Dict[str, int]:
    """
    Print and return the number of trainable
    and total parameters of a model.
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = getattr(model, "__name__", model.__class__.__name__)
    logging.info(f"Number of trainable/total parameters in {model_name}: " f"{num_trainable_params}/{num_params}")
    return {"trainable": num_trainable_params, "total": num_params}


def get_model_outputs_only(outputs: _TensorOrTensors) -> _TensorOrTensors:
    """
    Use this function to get just the raw
    outputs. Useful for many libraries, e.g.
    `transformers` and `allennlp` that return
    a tuple from the model, comprising loss,
    attention matrices, etc. too.
    """
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    return outputs


def send_model_to_device(model: nn.Module, device: _Device, device_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Send a model to specified device.
    Will also parallelize model if required.

    Note: `model.to()` is an inplace operation, so it will move the
          original model to the desired device. If the original model
          is to be retained on the original device, and a copy is
          to be moved to the desired device(s) and returned, make
          sure to set `model.copy()` to this function.
    """
    logging.info(f"Setting default device for model to {device}...")
    # Note: `model.to()` doesn't work as desired if model is
    # parallelized (model is still wrapped inside
    # `module`); therefore must do `model.module.to()`
    model = model.module.to(device) if hasattr(model, "module") else model.to(device)
    logging.info("Done.")

    # Set default value here instead of in signature
    # See: http://www.omahapython.org/IdiomaticPython.html#default-parameter-values
    if device_ids is None:
        device_ids = []

    # Parallelize model
    n_gpu = len(device_ids)
    if n_gpu > 1:
        logging.info(f"Using {n_gpu} GPUs: {device_ids}...")
        model = DataParallel(model, device_ids=device_ids)
        logging.info("Done.")
    return model


def send_batch_to_device(batch: _Batch, device: _Device, non_blocking: Optional[bool] = True) -> _Batch:
    """
    Send batch to given device.

    :param non_blocking: If True and this copy is between CPU
                         and GPU, the copy may occur asynchronously
                         with respect to the host. For other cases,
                         this argument has no effect.
                         For explanation, see: https://stackoverflow.com/a/55564072

    Useful when the batch tuple is of variable lengths.
    Specifically,
        - In regular multiclass setting:
            batch = (product_embedding, y)
        - In one-hot encoded multiclass / multilabel setting (e.g. ABSANet):
            batch = ( (product_embedding, label_embedding), y )
    This function will recursively send all tensors to the
    device retaining the original structure of the batch.

    E.g.:
        >>> a = torch.tensor([1,2,3], device="cpu")
        >>> b = torch.tensor([4,5,6], device="cpu")
        >>> c = torch.tensor([7,8,9], device="cpu")
        >>> batch = ((a, b), c)
        >>> cuda_batch = send_batch_to_device(batch, "cuda:0")
        >>> compare_tensors_or_arrays(cuda_batch, batch)
        True
        >>> is_batch_on_gpu(batch)
        False
        >>> is_batch_on_gpu(cuda_batch)
        True
    """
    if torch.is_tensor(batch):
        return batch.to(device=device, non_blocking=non_blocking)
    elif isinstance(batch, (list, tuple)):
        # Retain same data type as original
        return type(batch)(send_batch_to_device(e, device, non_blocking) for e in batch)
    else:  # Structure/type of batch unknown
        logging.warning(f"Type '{type(batch)}' not understood. Returning variable as-is.")
        return batch


def send_optimizer_to_device(optimizer: Optimizer, device: _Device) -> Optimizer:
    """
    Send an optimizer to specified device.
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return optimizer


def convert_tensor_to_numpy(batch: _Batch) -> _Batch:
    """
    Convert torch tensor(s) on any device to numpy array(s).
    Similar to `send_batch_to_device()`, can take a
    `torch.Tensor` or a tuple/list of them as input.
    """
    if torch.is_tensor(batch):
        return batch.to("cpu").detach().numpy()
    elif isinstance(batch, (list, tuple)):
        # Retain same data type as original
        return type(batch)(convert_tensor_to_numpy(e) for e in batch)
    else:  # Structure/type of batch unknown
        logging.warning(f"Type '{type(batch)}' not understood. Returning variable as-is.")
        return batch


def convert_numpy_to_tensor(
    batch: _Batch, device: Optional[_Device] = None, non_blocking: Optional[bool] = True
) -> _Batch:
    """
    Convert numpy array(s) to torch tensor(s) and
    optionally sends them to the desired device.
    Inverse operation of `convert_tensor_to_numpy()`,
    and similar to it, can take a np.ndarray or a
    tuple/list of them as input.
    """
    if isinstance(batch, np.ndarray):
        batch = torch.as_tensor(batch)
        return batch if device is None else send_batch_to_device(batch, device, non_blocking)
    elif isinstance(batch, (list, tuple)):
        # Retain same data type as original
        return type(batch)(convert_numpy_to_tensor(e, device, non_blocking) for e in batch)
    else:  # Structure/type of batch unknown
        logging.warning(f"Type '{type(batch)}' not understood. Returning variable as-is.")
        return batch


def compare_tensors_or_arrays(batch_a: _Batch, batch_b: _Batch) -> bool:
    """
    Compare the contents of two batches.
    Each batch may be of type `np.ndarray` or
    `torch.Tensor` or a list/tuple of them.

    Will return True if the types of the two
    batches are different but contents are the same.
    """
    if torch.is_tensor(batch_a):
        batch_a = convert_tensor_to_numpy(batch_a)
    if torch.is_tensor(batch_b):
        batch_b = convert_tensor_to_numpy(batch_b)

    if isinstance(batch_a, np.ndarray) and isinstance(batch_b, np.ndarray):
        return np.all(batch_a == batch_b)
    elif isinstance(batch_a, (list, tuple)) and isinstance(batch_b, (list, tuple)):
        return all(compare_tensors_or_arrays(a, b) for a, b in zip(batch_a, batch_b))
    else:  # Structure/type of batch unknown
        raise TypeError(
            f"Types of each batch '({type(batch_a)}, {type(batch_b)})' must "
            f"be `np.ndarray`, `torch.Tensor` or a list/tuple of them."
        )


def compare_model_parameters(parameters1: Iterable[torch.Tensor], parameters2: Iterable[torch.Tensor]) -> bool:
    """
    Compare two sets of model parameters.
    Useful in unit tests for ensuring consistency
    on saving and then loading the same set of
    parameters.
    """
    for p1, p2 in zip(parameters1, parameters2):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def compare_model_state_dicts(
    state_dict1: OrderedDict[str, _TensorOrTensors], state_dict2: OrderedDict[str, _TensorOrTensors]
) -> bool:
    """
    Compare two sets of model state dicts.
    Useful in unit tests for ensuring
    consistency on saving and then
    loading the same state dict.
    """
    for key1, key2 in zip(state_dict1, state_dict2):
        if state_dict1[key1].ne(state_dict2[key2]).sum() > 0:
            return False
    return True


def is_batch_on_gpu(batch: _Batch) -> bool:
    """
    Check if a `batch` is on a GPU.

    Similar to `send_batch_to_device()`, can take a
    `torch.Tensor` or a tuple/list of them as input.
    """
    if torch.is_tensor(batch):
        return batch.is_cuda
    elif isinstance(batch, (list, tuple)):
        return all(is_batch_on_gpu(e) for e in batch)
    else:  # Structure/type of batch unknown
        raise TypeError(f"Type '{type(batch)}' not understood.")


def is_model_on_gpu(model: nn.Module) -> bool:
    """
    Check if a `model` is on a GPU.
    """
    return is_batch_on_gpu(next(model.parameters()))


def is_model_parallelized(model: nn.Module) -> bool:
    """
    Check if a `model` is parallelized on multiple GPUs.
    """
    return is_model_on_gpu(model) and isinstance(model, DataParallel)


def get_total_grad_norm(parameters: Iterable[torch.Tensor], norm_type: Optional[float] = 2) -> torch.Tensor:
    """
    Get the total `norm_type` norm
    over all parameter gradients.
    """
    return nn.utils.clip_grad_norm_(parameters, max_norm=np.inf, norm_type=norm_type)


def get_model_performance_trackers(config: _Config) -> Tuple[ModelTracker, ModelTracker]:
    """
    Initialize loss and eval criteria
    loggers for train and val datasets.
    """
    train_logger = ModelTracker(config, is_train=True)
    val_logger = ModelTracker(config, is_train=False)
    return train_logger, val_logger


class ModelTracker:
    """
    Class for tracking model's progress.

    Use this for keeping track of the loss and
    any evaluation metrics (accuracy, f1, etc.)
    at each epoch.
    """

    def __init__(self, config: _Config, is_train: Optional[bool] = True):
        self.eval_criteria = config.eval_criteria
        self.is_train = is_train
        if not is_train:
            self.early_stopping_criterion = config.early_stopping_criterion
        self._init_progress_trackers()

    def _init_progress_trackers(self):
        """
        Initialize the loss/eval_criteria tracking dictionaries.
        """
        self.loss_hist, self.eval_metrics_hist = OrderedDict(), OrderedDict()
        for eval_criterion in self.eval_criteria:
            self.eval_metrics_hist[eval_criterion] = OrderedDict()

    def add_losses(self, losses: List[float], epoch: Optional[int] = -1) -> None:
        """
        Store the losses at a given epoch.
        :param epoch: If not provided, will store
                      at the next epoch.
        """
        epoch = self._get_next_epoch(epoch, "loss")
        if not isinstance(losses, list):
            losses = [losses]
        self.loss_hist[epoch] = losses

    def get_losses(
        self, epoch: Optional[int] = None, flatten: Optional[bool] = False
    ) -> Union[List[float], OrderedDict[str, List[float]]]:
        """
        Get the loss history.
        :param epoch: If provided, returns the list
                      of losses at that epoch,
                      otherwise the whole dictionary.
                      If epoch=-1, returns list of
                      losses at last epoch.
        :param flatten: If true, a single list of all
                        flattened values is returned.
        """
        epoch = self._get_correct_epoch(epoch, "loss")
        if epoch is not None:
            return self.loss_hist[epoch]
        if flatten:  # Flatten across all epochs
            return self.get_all_losses()
        return self.loss_hist

    def get_all_losses(self) -> List[float]:
        """
        Get the entire loss history across all
        epochs flattened into one list.
        """
        return np.concatenate(list(self.loss_hist.values())).tolist()

    def add_eval_metrics(self, eval_metrics: Dict[str, float], epoch: Optional[int] = -1) -> None:
        """
        Store the eval_metrics at a given epoch.
        :param epoch: If not provided, will store
                      at the next epoch.
        """
        epoch = self._get_next_epoch(epoch, "eval_metrics")
        for eval_criterion in self.eval_criteria:
            self.eval_metrics_hist[eval_criterion][epoch] = eval_metrics[eval_criterion]

    def get_eval_metrics(
        self, eval_criterion: Optional[str] = None, epoch: Optional[int] = None, flatten: Optional[bool] = False,
    ) -> Union[float, List[float], OrderedDict[str, Union[float, List[float]]]]:
        """
        Get the evaluation metrics history.
        :param eval_criterion: The criterion whose history
                               is to be returned.
        :param epoch: The epoch for which the history
                      is to be returned.
        - If both params are provided, the value at that epoch
          is returned.
        - If only eval_criterion is provided:
            - If `flatten=False`, a dictionary of values
              at each epoch is returned
            - If `flatten=True`, the values across all
              epochs are flattened into a single list
        - If only `epoch` is provided, a dictionary of values
          for each criterion at that epoch is returned.
        If `epoch=-1`, returns list of losses at last epoch.
        """
        epoch = self._get_correct_epoch(epoch, "eval_metrics")
        if eval_criterion is not None:
            if epoch is not None:  # Both params provided
                return self.eval_metrics_hist[eval_criterion][epoch]
            elif flatten:  # Flatten across all epochs
                return self.get_all_eval_metrics(eval_criterion)
            return self.eval_metrics_hist[eval_criterion]  # Return ordered dict
        elif epoch is not None:
            return OrderedDict(
                {
                    eval_criterion: self.eval_metrics_hist[eval_criterion][epoch]
                    for eval_criterion in self.eval_criteria
                }
            )
        return self.eval_metrics_hist

    def get_all_eval_metrics(self, eval_criterion: Optional[str] = None) -> Union[List[float], Dict[str, List[float]]]:
        """
        Get the entire eval_metrics history across all
        epochs flattened into one list for each eval_criterion.
        :param eval_criterion: If provided, only the list of
                               history for that eval_criterion
                               is returned.
        """

        def get_eval_criterion_metrics(eval_criterion):
            return list(self.eval_metrics_hist[eval_criterion].values())

        if eval_criterion is not None:
            return get_eval_criterion_metrics(eval_criterion)
        eval_metrics_dict = OrderedDict(
            {eval_criterion: get_eval_criterion_metrics(eval_criterion) for eval_criterion in self.eval_criteria}
        )
        return eval_metrics_dict

    def log_epoch_metrics(self, epoch: Optional[int] = -1) -> str:
        """
        Log loss and evaluation metrics for a
        given epoch in the following format:
        "TRAIN Epoch: 1  Average loss: 0.5, ACCURACY: 0.8, PRECISION: 0.7"
        """
        epoch_loss = self._get_correct_epoch(epoch, "loss")
        epoch_eval_metrics = self._get_correct_epoch(epoch, "eval_metrics")
        assert epoch_loss == epoch_eval_metrics
        dataset_type = "TRAIN" if self.is_train else "VAL  "
        mean_loss_epoch = np.mean(self.get_losses(epoch=epoch_loss))
        result_str = f"\n\033[1m{dataset_type} Epoch: {epoch_loss}" f"\tAverage loss: {mean_loss_epoch:.4f}, "
        result_str += ", ".join(
            [
                f"{eval_criterion}: {self.get_eval_metrics(eval_criterion, epoch_loss):.4f}"
                for eval_criterion in self.eval_criteria
            ]
        )
        result_str += "\033[0m\n"
        logging.info(result_str)
        return result_str

    def add_metrics(self, losses: List[float], eval_metrics: Dict[str, float], epoch: Optional[int] = -1) -> None:
        """
        Shorthand function to add losses
        and eval metrics at the end of
        a given epoch.
        """
        self.add_losses(losses, epoch)
        self.add_eval_metrics(eval_metrics, epoch)

    def add_and_log_metrics(
        self, losses: List[float], eval_metrics: Dict[str, float], epoch: Optional[int] = -1
    ) -> str:
        """
        Shorthand function to add losses
        and eval metrics at the end of a
        given epoch, and then print the
        results for that epoch.
        """
        self.add_metrics(losses, eval_metrics, epoch)
        return self.log_epoch_metrics(epoch)

    def get_early_stopping_metric(self) -> float:
        """
        For validation loggers, returns the
        `early_stopping_criterion` for the
        last epoch for which history is stored.
        """
        if self.is_train:
            raise ValueError("Early stopping must be applied on validation set.")
        return self.get_eval_metrics(self.early_stopping_criterion, -1)

    def get_eval_metrics_df(self, epoch: Optional[int] = None) -> pd.DataFrame:
        """
        Get a DataFrame object of all eval metrics
        for all (or optionally a specific) epoch(s).
        """
        metrics_df = pd.DataFrame.from_dict(self.get_eval_metrics())
        metrics_df.insert(loc=0, column="epoch", value=metrics_df.index)
        metrics_df.reset_index(drop=True, inplace=True)
        if epoch is not None:
            epoch = self._get_correct_epoch(epoch, "loss")
            return metrics_df.query("epoch == @epoch")
        return metrics_df

    def set_best_epoch(self, best_epoch: Optional[int] = None) -> None:
        """
        Add the `best_epoch` attribute to validation
        logger for future evaluation purposes.
        """
        if self.is_train:
            raise ValueError("Best epoch can only be stored into validation logger.")
        if best_epoch is None:
            self.best_epoch = self.get_overall_best_epoch()
        else:
            if best_epoch not in self.epochs:
                raise ValueError(f"Best epoch provided ({best_epoch}) must be one of {self.epochs}.")
            self.best_epoch = best_epoch

    def get_overall_best_epoch(self) -> int:
        """
        Get the overall best epoch if early stopping is not used.

        Returns the maximum value across all epochs based
        on the (early) stopping criterion, which defaults
        to accuracy / mse if it isn't defined.
        """
        eval_metrics_dict = self.get_eval_metrics(self.early_stopping_criterion)
        best_epoch = max(eval_metrics_dict, key=eval_metrics_dict.get)
        return best_epoch

    @property
    def _epochs_loss(self) -> List[int]:
        """
        List of epochs for which loss history is stored.
        """
        return list(self.loss_hist.keys())

    @property
    def _epochs_eval_metrics(self) -> List[int]:
        """
        List of epochs for which eval metrics history is stored.
        """
        k = list(self.eval_metrics_hist.keys())[0]  # Any random metric
        return list(self.eval_metrics_hist[k].keys())

    @property
    def epochs(self) -> List[int]:
        """
        Returns the total list of epochs for which history is stored.

        Assumes that history is stored for the same number of epochs
        for both loss and eval_metrics.
        """
        assert self._epochs_loss == self._epochs_eval_metrics
        return self._epochs_loss

    def _get_correct_epoch(self, epoch: int, hist_type: str) -> int:
        """
        If `epoch=-1`, returns the last epoch for
        which history is currently stored, otherwise
        the epoch itself.
        """
        if epoch == -1:
            total_epochs = self._epochs_loss if hist_type == "loss" else self._epochs_eval_metrics
            return max(total_epochs) if len(total_epochs) else 0
        return epoch

    def _get_next_epoch(self, epoch: int, hist_type: str) -> int:
        """
        If `epoch=-1`, returns the next epoch for
        which history is to be stored, otherwise
        the epoch itself.
        """
        if epoch == -1:
            total_epochs = self._epochs_loss if hist_type == "loss" else self._epochs_eval_metrics
            epoch = max(total_epochs) if len(total_epochs) else 0
        return epoch + 1


class SequencePooler(nn.Module):
    """
    Pool the sequence output for transformer-based models.

    Class used instead of lambda functions to remain
    compatible with `torch.save()` and `torch.load()`.
    """

    DEFAULT_POOLER_TYPE = "default"

    def __init__(self, model_type: Optional[str] = "bert"):
        """
        :param model_type: Type of `transformers` model.
                           Can be manually specified or extracted
                           from the model class like this:
                               >>> from transformers import AutoModel
                               >>> model = AutoModel.from_pretrained("roberta-base")
                               >>> model.config.model_type
                               "roberta"
        """
        super().__init__()
        self._set_pooler(model_type)

    def __repr__(self):
        return f"{self.__class__.__name__}(model_type={self.model_type})"

    def forward(self, x):
        return self.pooler(x)

    def _set_pooler(self, model_type: str) -> None:
        """
        Set the appropriate pooler as per the `model_type`.
        """
        # Set the appropriate pooler as per `model_type`
        self.POOLER_MAPPING = {
            "bert": self._bert_pooler,
            "distilbert": self._distilbert_pooler,
            "albert": self._albert_pooler,
            "roberta": self._roberta_pooler,
            "electra": self._electra_pooler,
        }

        # Use default pooler if not supported
        if model_type in self.POOLER_MAPPING.keys():
            self.model_type = model_type
            self.pooler = self.POOLER_MAPPING[self.model_type]
        else:
            logging.warning(
                f"No supported sequence pooler was found for model of " f"type '{model_type}'. Using the default one."
            )
            self.model_type = self.DEFAULT_POOLER_TYPE
            self.pooler = self._default_pooler

    def _default_pooler(self, x):
        return x

    def _bert_pooler(self, x):
        """
        **NOTE**: The sentence/sequence vector obtained
        from BERT does NOT correspond to the [CLS] vector.
        It takes as input this vector and then runs a small
        network on top of it to give the "pooled" sequence output.
        See:
        1. https://github.com/huggingface/transformers/blob/1cdd2ad2afb73f6af185aafecb7dd7941a90c4d1
           /src/transformers/modeling_bert.py#L426-L438
        2. https://github.com/huggingface/transformers/blob/1cdd2ad2afb73f6af185aafecb7dd7941a90c4d1
           /src/transformers/modeling_bert.py#L738-L739
        3. https://www.kaggle.com/questions-and-answers/86510
        """
        return x[1]  # Pooled seq vector

    def _distilbert_pooler(self, x):
        return x[0][:, 0]  # [CLS] vector

    def _albert_pooler(self, x):
        return self._bert_pooler(x)  # Same as BERT (see above)

    def _roberta_pooler(self, x):
        return x[0][:, 0]  # <s> vector (equiv. to [CLS])

    def _electra_pooler(self, x):
        return x[0][:, 0]  # [CLS] vector


class DataParallel(nn.DataParallel):
    """
    Custom DataParallel class inherited from `nn.DataParallel`.

    Purpose is to allow direct access to model attributes and
    methods when it is wrapped in a `module` attribute because
    of `nn.DataParallel`.
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def __getattr__(self, name):
        """
        Return model's own attribute if available, otherwise
        fallback to attribute of parent class.

        Solves the issue that when `nn.DataParallel` is applied,
        methods and attributes defined in `BasePyTorchModel`
        like `predict()` can only be accessed with
        `self.module.predict()` instead of `self.predict()`.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DaskProgressBar(Callback):
    """
    Real-time tqdm progress bar adapted to dask dataframes (for `apply`).
    Code reference: https://github.com/tqdm/tqdm/issues/278#issue-180452055
    """

    def _start_state(self, dsk, state):
        self._tqdm = tqdm(total=sum(len(state[k]) for k in ["ready", "waiting", "running", "finished"]))

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        pass


class GELU(nn.Module):
    """
    Implementation of the gelu activation function
    currently in Google BERT repo (identical to OpenAI GPT).
    Also see: https://arxiv.org/abs/1606.08415

    Code reference:
    https://github.com/huggingface/transformers/blob/1cdd2ad2afb73f6af185aafecb7dd7941a90c4d1
    /src/transformers/activations.py#L25-L29
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
