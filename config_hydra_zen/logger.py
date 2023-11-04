from hydra_zen import store
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from .utils import full_builds


logger_store = store(group='trainer/logger')
logger_store(
    full_builds(
        TensorBoardLogger,
        save_dir='${hydra:runtime.output_dir}',
        name='tb',
        version='',
        log_graph=True
    ),
    name='TensorBoardLogger'
)
