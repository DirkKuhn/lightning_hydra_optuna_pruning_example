# Deep-learning example for OptunaPruningSweeper

This example is adapted from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py.
It shows how more complicated experiments can be configured with [hydra](https://hydra.cc/) or
[hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/).
[Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/) is used to avoid boilerplate for
training the neural networks. This example also shows how the
[``OptunaPruningSweeper``]
can be used.

To show a more complicated configuration a ResNet model is implemented adapted from
[``torchvision``](https://pytorch.org/vision/stable/models/resnet.html). However, pre-activation
ResNet blocks are used. The
[``FashionMNIST``](https://pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html#torchvision.datasets.FashionMNIST)
dataset is used.

The hydra specific code/configuration is located in ``run_hydra`` and ``config_hydra``.
The code/configuration specific to hydra_zen can be found in ``run_hydra_zen`` and ``config_hydra_zen``.

## Installation

It is recommended to first create a virtual environment. In this environment install the dependencies with
```
pip install -r requirements.txt
```
To run the hyperparameter optimization with pruning install the ``OptunaPruningSweeper``. This plugin has not yeet
been added to PyPI. Install it by cloning the repository ... and execute in your virtual environment
```
pip install PATH-TO-CLONED-REPOSITORY-OF-HYDRA-OPTUNA-PRUNING-SWEEPER
```

If you are interested in a template for pytorch-lightning + hydra also take a look at
https://github.com/ashleve/lightning-hydra-template.