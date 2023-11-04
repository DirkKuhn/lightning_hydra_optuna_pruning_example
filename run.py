"""
In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch Lightning, and FashionMNIST. We optimize the neural network architecture.
"""
import logging
from typing import Optional, Literal

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner


log = logging.getLogger(__name__)


def train_nn(
        trainer: pl.Trainer,
        task: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        compile_mode: Literal[False] | str,
        auto_lr_find: bool,
        monitor: str,
        test: bool
) -> Optional[float]:
    if compile_mode != False:
        log.info('Compiling model')
        task = th.compile(task, mode=compile_mode)

    if auto_lr_find:
        log.info('Automatically determine learning rate')
        lr = find_lr(trainer, task, datamodule)
        log.info(f'Learning rate set to {lr}')

    log.info('Start training')
    trainer.fit(task, datamodule=datamodule)

    if test:
        log.info('Start testing')
        trainer.test(datamodule=datamodule, ckpt_path='best')
    else:
        log.info('Start validating')
        val_metrics = trainer.validate(datamodule=datamodule, ckpt_path='best')
        return val_metrics[0][f'val/{monitor}']


def find_lr(
        trainer: pl.Trainer, task: pl.LightningModule, dm: pl.LightningDataModule,
        min_lr: float = 1e-5, max_lr: float = 1, plot: bool = True
) -> float:
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(task, datamodule=dm, min_lr=min_lr, max_lr=max_lr, early_stop_threshold=None)
    if plot:
        lr_finder.plot(suggest=True).show()
    lr = lr_finder.suggestion()
    return lr
