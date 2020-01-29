# Overview

This reposity contains PyTorch code that is common and (hopefully) helpful to most projects built on PyTorch.

Please refer to [this document](https://docs.google.com/presentation/d/1mAa8TetcDcjckezrWywpA8XhUIyxnv5Wnr-mSchF8S0/edit?usp=sharing) to see what functionalities it provides.

In a nutshell, it has code for:
  - Training / testing models
  - Logging all common losses / eval metrics
  - `BasePyTorchModel`, which has:
    - `initialize_model()`:
      - Prints number of params + architecture
      - Allows initializing (all / given) weights for Conv, BatchNorm, Linear, Embedding layers
    - Provision to freeze (all / given) weights of model
  - Sending model to device(s)
  - Saving/loading/removing state dict / model checkpoints
  - Early stopping
  - Properly sending model/optimizer/batch to device(s)
  - Defining custom train/test loss and evaluation criteria directly from config
    - Supports all common losses/metrics for regression and binary/multi-class/multi-label classification
    - May give as many as you like
  - Cleanly stopping training at any point without losing progress
  - Make predictions
    - Labels and probabilities for classification, with an option to set the threshold probability
    - Raw outputs for regression
  - Loading back best (or any) model and printing + plotting all losses + eval metrics
  - etc.


# Usage
I'll be adding example scripts soon. For the time being, you may refer to [this repo](https://gitlab.dev.tripadvisor.com/vrds/exp_vi_predict_product_subcats/tree/initial) which leverages `pytorch_common`.


# Disclaimer

This repo is a personal project, and as such, has not been as heavily tested. It is (and will likely always be) a work-in-progress, as I try my best to keep it current with the advancements in PyTorch.

If you come across any bugs, or have questions/suggestions, please consider opening an issue, [reaching out to me](mailto:ranamihir@gmail.com), or better yet, sending across a PR. :)

Author: [Mihir Rana](https://github.com/ranamihir)