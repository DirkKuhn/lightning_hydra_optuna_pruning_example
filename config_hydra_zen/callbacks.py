from hydra_zen import store, builds
from hydra_plugins.hydra_optuna_pruning_sweeper import trial_provider
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from optuna.integration import PyTorchLightningPruningCallback
from .utils import full_builds


def inject_trial(cls):
    def wrapper(*args, **kwargs):
        trial = trial_provider.trial
        assert trial is not None
        return cls(*args, trial=trial_provider.trial, **kwargs)
    return wrapper


ModelCheckpointConf = full_builds(
    ModelCheckpoint,
    dirpath='${hydra:runtime.output_dir}/checkpoints',
    filename='{epoch}_{step}',
    monitor='val/${monitor}',
    save_top_k=1,
    mode='${mode}',
    every_n_epochs=1,
    save_last=False
)
ModelSummaryConf = full_builds(
    ModelSummary,
    max_depth=3
)
PyTorchLightningPruningCallbackConf = builds(
    PyTorchLightningPruningCallback,
    monitor='val/${monitor}',
    zen_wrappers=inject_trial
)

callbacks_store = store(group='trainer/callbacks')
callbacks_store(
    [ModelSummaryConf, ModelCheckpointConf],
    name='default'
)
callbacks_store(
    [PyTorchLightningPruningCallbackConf, ModelSummaryConf, ModelCheckpointConf],
    name='pruning'
)
