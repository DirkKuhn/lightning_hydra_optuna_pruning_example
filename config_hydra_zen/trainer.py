from hydra_zen import store
from pytorch_lightning import Trainer

trainer_store = store(group='trainer')
trainer_store(Trainer)
