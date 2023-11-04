from hydra_zen import store, MISSING

from run import train_nn
from .utils import full_builds


Config = full_builds(
    train_nn,
    hydra_defaults=[
        '_self_',
        {'trainer': 'Trainer'},
        {'trainer/logger': 'TensorBoardLogger'},
        {'task': 'ClassificationTask'},
        {'task/model': 'ResNet'},
        {'datamodule': 'FashionMNISTDataModule'},
        {'experiment': 'hparam_search'}
    ],
    datamodule=dict(
        val_fraction=0.1
    ),
    monitor='MulticlassAccuracy',
    zen_meta=dict(
        seed=MISSING,
        mode='max',
        num_classes=10
    )
)
store(Config, name='train_nn')
