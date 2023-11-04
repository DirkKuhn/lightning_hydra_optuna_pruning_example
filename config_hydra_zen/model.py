from hydra_zen import store, just
from models import ResNet, PreActResNetBlock, PreActBottleneckResNetBlock

model_store = store(group='task/model')
model_store(
    ResNet,
    num_classes='${num_classes}'
)

block_store = store(group='task/model/block')
block_store(just(PreActResNetBlock), name='PreActResNetBlock')
block_store(just(PreActBottleneckResNetBlock), name='PreActBottleneckResNetBlock')

