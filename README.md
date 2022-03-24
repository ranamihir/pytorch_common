
# PyTorch-Common
<p>
    <a href="https://github.com/ranamihir/pytorch_common/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/ranamihir/pytorch_common.svg">
    </a>
    <a href="https://github.com/ranamihir/pytorch_common/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/ranamihir/pytorch_common.svg">
    </a>
</p>


# Overview

This repository contains PyTorch code that is common and (hopefully) helpful to most projects built on PyTorch.

It is a lightweight wrapper that contains PyTorch code that is common and (hopefully) helpful to most projects built on PyTorch. It is built with 3 main principles in mind:
- Make use of PyTorch available to people without much in-depth knowledge of it while providing enormous flexibility and support for hardcore users
- Under-the-hood optimization for fast and memory efficient performance
- Ability to change all settings (e.g. model, loss, metrics, devices, hyperparameters, artifact directories, etc.) directly from config


# Features

In a nutshell, it has code for:
  - Training / testing models
    - Option to retrain on all data (without performing evaluation on a separate data set)
  - Logging all common losses / eval metrics
  - `BasePyTorchDataset`, which has functions for:
    - Printing summary + useful statistics
    - Over- / under-sampling rows
    - Properly saving / loading / removing datasets (using appropriate pickle modules)
  - `BasePyTorchModel`, which has:
    - `initialize_model()`:
      - Prints number of params + architecture
      - Allows initializing (all / given) weights for Conv, BatchNorm, Linear, Embedding layers
    - Provision to freeze / unfreeze (all / given) weights of model
  - Sending model to device(s)
  - Saving / loading / removing / copying state dict / model checkpoints
  - Disable above mentioned checkpointing from config for faster development
  - Early stopping
  - Sample weighting
  - Properly sending model / optimizer / batch to device(s)
  - Defining custom train / test loss and evaluation criteria directly from config
    - Supports most common losses / metrics for regression and binary / multi-class / multi-label classification
    - May give as many as you like
  - Cleanly stopping training at any point without losing progress
  - Make predictions
    - Labels and probabilities for classification, with an option to set the threshold probability
    - Raw outputs for regression
  - Loading back best (or any) model and printing + plotting all losses + eval metrics
  - etc.


# Installation
To install this package, you must have [pytorch](https://pytorch.org/) (and [transformers](https://github.com/huggingface/transformers) for accessing NLP-based functionalities) installed. Then you can simply install this package from source:
```bash
git clone git@github.com:ranamihir/pytorch_common.git
cd pytorch_common
conda env create -f requirements.yaml  # If you don't already have a pytorch-enabled conda environment
conda activate pytorch_common  # <-- Replace with your environment name
pip install .
```
which will create an environment called `pytorch_common` for you with all the required dependencies and this package installed.

If you'd like access to the NLP-related functionalities (specifically for [transformers](https://github.com/huggingface/transformers/)), make sure to install it as below instead:
```bash
pip install ".[nlp]"
```


# Usage

Training a very simple (dummy) model is as easy as:

```python
from torch.utils.data import DataLoader

from pytorch_common.config import load_pytorch_common_config
from pytorch_common.metrics import get_loss_eval_criteria
from pytorch_common.train_utils import train_model
from pytorch_common.utils import get_model_performance_trackers

# Load default pytorch_common config and override with your settings
project_config_dict = ...
config = load_pytorch_common_config(project_config_dict)

# Create your own training objects here
train_loader = ...
val_loader = ...
model = ...
optimizer = ...

# Use `pytorch_common` to get loss / eval criteria, initialize loggers, and train the model
loss_criterion_train, loss_criterion_eval, eval_criteria = get_loss_eval_criteria(config, reduction="mean")
train_logger, val_logger = get_model_performance_trackers(config)
return_dict = train_model(
    model, config, train_loader, val_loader, optimizer, loss_criterion_train, loss_criterion_eval, eval_criteria, train_logger, val_logger
)
```

More detailed examples highlighting the full functionality of this package can be found in the [examples](https://github.com/ranamihir/pytorch_common/tree/master/examples) directory.

## Config

A powerful advantage of using this repository is the ability to change a large number of settings related to PyTorch, and more generally, deep learning, directly from YAML, instead of having to worry about making code changes.

To do this, all you need to do is invoke the `load_pytorch_common_config()` function (with your project dictionary as input, if required). This will allow you to edit all `pytorch_common` supported settings in your project dictionary / YAML, or use the default ones for those not specified. E.g.:

```python
>>> from pytorch_common.config import load_pytorch_common_config

>>> config = load_pytorch_common_config()  # Use default settings
>>> print(config.batch_size_per_gpu)
32
>>> dictionary = {"vocab_size": 10_000, "batch_size_per_gpu": 64}  # Override default settings and / or add project specific settings here
>>> config = load_pytorch_common_config(dictionary)
>>> print(config.batch_size_per_gpu)
64
>>> print(config.vocab_size)
10000
```

The list of all supported configuration settings can be found [here](https://github.com/ranamihir/pytorch_common/blob/master/pytorch_common/configs/config.yaml).


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

If you come across any bugs, or have questions / suggestions, please consider opening an issue, [reaching out to me](mailto:ranamihir@gmail.com), or better yet, sending across a PR. :)

Author: [Mihir Rana](https://github.com/ranamihir)
