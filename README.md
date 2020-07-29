
# PyTorch-Common
<p>
    <a href="https://github.com/ranamihir/pytorch_common/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ranamihir/pytorch_common.svg">
    </a>
    <a href="https://github.com/ranamihir/pytorch_common/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/ranamihir/pytorch_common.svg">
    </a>
</p>

`pytorch-common` is a lightweight wrapper that contains PyTorch code that is common and (hopefully) helpful to most projects built on PyTorch. It is built with 3 main principles in mind:
- Make use of PyTorch available to people without much in-depth knowledge of it while providing enormous flexibility and support for hardcore users
- Under-the-hood optimization for fast and memory efficient performance
- Ability to change all settings (e.g. model, loss, metrics, devices, hyperparameters, artifact directories, etc.) directly from config

# Features

In a nutshell, it has code for:
  - Training / testing models
  - Logging all common losses / eval metrics
  - `BasePyTorchDataset`, which has functions for:
    - Printing summary + useful statistics
    - Over-/under-sampling rows
    - Properly saving/loading/removing datasets (using appropriate pickle modules)
  - `BasePyTorchModel`, which has:
    - `initialize_model()`:
      - Prints number of params + architecture
      - Allows initializing (all / given) weights for Conv, BatchNorm, Linear, Embedding layers
    - Provision to freeze/unfreeze (all / given) weights of model
  - Sending model to device(s)
  - Saving/loading/removing/copying state dict / model checkpoints
  - Disable above mentioned checkpointing from config for faster development
  - Early stopping
  - Properly sending model/optimizer/batch to device(s)
  - Defining custom train/test loss and evaluation criteria directly from config
    - Supports most common losses/metrics for regression and binary/multi-class/multi-label classification
    - May give as many as you like
  - Cleanly stopping training at any point without losing progress
  - Make predictions
    - Labels and probabilities for classification, with an option to set the threshold probability
    - Raw outputs for regression
  - Loading back best (or any) model and printing + plotting all losses + eval metrics
  - etc.

# Installation
To install this package, you must have [pytorch](https://pytorch.org/) (and [transformers](https://github.com/huggingface/transformers) for accessing NLP-based functionalities) installed.
If you don't already have it, you can create a conda environment by running:
```bash
conda env create -f requirements.yaml`
pip install -e . # or ".[nlp]" if required
```
which will create an environment called `pytorch_common` for you with all the required dependencies.


The package can then be installed from source:
```bash
git clone git@github.com:ranamihir/pytorch_common
cd pytorch_common
pip install .
```

If you'd like access to the NLP-related functionalities (specifically for [transformers](https://github.com/huggingface/transformers/)), make sure to install it as below instead:
```bash
pip install ".[nlp]"
```

# Usage

The default [config](https://github.com/ranamihir/pytorch_common/blob/master/pytorch_common/configs/config.yaml) can be loaded, and overridden with a user-specified dictionary, as follows:
```python
from pytorch_common.config import load_pytorch_common_config

# Create your own config (or load from a yaml file)
config_dict = {"batch_size_per_gpu": 5, "device": "cpu", "epochs": 2, "lr": 1e-3, "disable_checkpointing": True}

# Load the deault pytorch_common config, and then override it with your own custom one
config = load_pytorch_common_config(config_dict)
```

Then, training a (dummy) model is as easy as:
```python
from torch.utils.data import DataLoader
from torch.optim import SGD

from pytorch_common.additional_configs import BaseDatasetConfig, BaseModelConfig
from pytorch_common.datasets import create_dataset
from pytorch_common.metrics import get_loss_eval_criteria
from pytorch_common.models import create_model
from pytorch_common.train_utils import train_model
from pytorch_common.utils import get_model_performance_trackers

# Create your own objects here
dataset_config = BaseDatasetConfig({"size": 5, "dim": 1, "num_classes": 2})
model_config = BaseModelConfig({"in_dim": 1, "num_classes": 2})
dataset = create_dataset("multi_class_dataset", dataset_config)
train_loader = DataLoader(dataset, batch_size=config.train_batch_size)
val_loader = DataLoader(dataset, batch_size=config.eval_batch_size)
model = create_model("single_layer_classifier", model_config)
optimizer = SGD(model.parameters(), lr=config.lr)

# Use `pytorch_common` to get loss/eval criteria, initialize loggers, and train the model
loss_criterion_train, loss_criterion_eval, eval_criteria = get_loss_eval_criteria(config, reduction="mean")
train_logger, val_logger = get_model_performance_trackers(config)
return_dict = train_model(
    model, config, train_loader, val_loader, optimizer, loss_criterion_train, loss_criterion_eval, eval_criteria, train_logger, val_logger
)
```
For more details on getting started, check out the [basic usage notebook](https://github.com/ranamihir/pytorch_common/blob/master/notebooks/basic_usage.ipynb) and other examples in the [notebooks](https://github.com/ranamihir/pytorch_common/blob/master/notebooks/) folder.

# Testing

Several unit tests are present in the [tests](https://github.com/ranamihir/pytorch_common/tree/master/tests) directory. You may manually run them with:

```bash
python -m unittest discover tests
```

Make sure to first activate the environment that has [pytorch](https://pytorch.org/) (and optionally [transformers](https://github.com/huggingface/transformers)) installed.

Additionally, I have also added a pre-push hook so that all tests are run locally before each push.
This is done because these tests take some time (depending on resources available), and pre-commit hooks tend to slow down development.
To install the pre-push hook, just run:

```bash
chmod +x install-hooks.sh
./install-hooks.sh
```

In the future, I intend to move the tests to CI.

# To-do's

I have some enhancements in mind which I haven't gotten around to adding to this repo yet:
  - Adding automatic mixed precision training (AMP) to enable it directly from config
  - Enabling distributed training across servers


# Disclaimer

This repo is a personal project, and as such, has not been as heavily tested. It is (and will likely always be) a work-in-progress, as I try my best to keep it current with the advancements in PyTorch.

If you come across any bugs, or have questions/suggestions, please consider opening an issue, [reaching out to me](mailto:ranamihir@gmail.com), or better yet, sending across a PR. :)

Author: [Mihir Rana](https://github.com/ranamihir)
