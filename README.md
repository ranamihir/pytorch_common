This reposity contains PyTorch code that is (hopefully) common and helpful to most projects built on PyTorch.

Please refer to [this document](https://docs.google.com/presentation/d/1mAa8TetcDcjckezrWywpA8XhUIyxnv5Wnr-mSchF8S0/edit?usp=sharing) to see what functionalities it provides.

In a nutshell, it has code for:
  - Training / testing models
  - Logging all common losses / eval metrics
  - `BasePyTorchModel`, which has:
    - `initialize_model()`:
      - Prints number of params + architecture
      - Sets prediction function (classification / regression)
      - Allows initializing (all / given) weights for Conv, BatchNorm, Linear, Embedding layers
    - Provision to freeze (all / given) weights of model
  - Sending model to device(s)
  - Saving/loading/removing state dict / model checkpoints
  - Early stopping
  - Properly sending model/optimizer/batch to device(s)
  - Defining custom train/test loss and evaluation criteria directly from config
    - Supports all common losses/metrics for binary/multi-class/multi-label
    - May give as many as you like
  - Cleanly stopping training at any point without losing progress
  - Loading back best (or any) model and printing + plotting all losses + eval metrics
  - etc.
