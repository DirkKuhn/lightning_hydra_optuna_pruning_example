from typing import Optional, Literal

import pytorch_lightning as pl
import hydra
from hydra.utils import instantiate
from hydra_plugins.hydra_optuna_pruning_sweeper import trial_provider
from omegaconf import DictConfig
from optuna import Trial

from run import train_nn


@hydra.main(version_base='1.3', config_path='config_hydra', config_name='config')
def hydra_train_nn(cfg: DictConfig) -> Optional[float]:
    if cfg.get('seed'):
        pl.seed_everything(cfg.seed, workers=True)

    # Inject trial in `PyTorchLightningPruningCallback`
    callbacks = []
    for cb_cfg in cfg.trainer.callbacks:
        if 'PruningCallback' in cb_cfg._target_:
            trial = trial_provider.trial
            assert isinstance(trial, Trial)
            cb = instantiate(cb_cfg, trial=trial)
        else:
            cb = instantiate(cb_cfg)
        callbacks.append(cb)
    del cfg.trainer._callback_dict

    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks)
    task: pl.LightningModule = instantiate(cfg.task)
    datamodule: pl.LightningDataModule = instantiate(cfg.datamodule)

    compile_mode: Literal[False] | str = cfg.compile_mode
    auto_lr_find: bool = cfg.auto_lr_find
    monitor: str = cfg.monitor
    test: bool = cfg.test

    return train_nn(
        trainer=trainer, task=task, datamodule=datamodule,
        compile_mode=compile_mode, auto_lr_find=auto_lr_find,
        monitor=monitor, test=test
    )


if __name__ == '__main__':
    hydra_train_nn()
