import os
from hydra_zen import store
from datamodules import FashionMNISTDataModule
from .utils import full_builds

datamodule_store = store(group='datamodule')
datamodule_store(
    FashionMNISTDataModule,
    data_dir=full_builds(os.getcwd)
)
