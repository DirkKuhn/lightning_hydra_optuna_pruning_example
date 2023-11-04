from hydra_zen import store
from torch import optim


optimizer_store = store(group='task/optimizer')
optimizer_store(
    optim.SGD,
    zen_partial=True
)
optimizer_store(
    optim.AdamW,
    zen_partial=True
)
