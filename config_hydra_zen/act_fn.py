from hydra_zen import store, just
from torch import nn

act_fn_store = store(group='task/model/act_fn')
act_fn_store(just(nn.ReLU), name='ReLU')
act_fn_store(just(nn.LeakyReLU), name='LeakyReLU')
act_fn_store(just(nn.SELU), name='SELU')
act_fn_store(just(nn.Mish), name='Mish')
