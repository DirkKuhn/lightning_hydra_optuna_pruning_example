from hydra_zen import just, make_config
from torch import nn
from torchvision import transforms

from models import PreActResNetBlock, PreActBottleneckResNetBlock
from ..config import Config
from ..utils import full_builds, experiment_store


experiment_store(
    make_config(
        hydra_defaults=[
            {'/trainer/callbacks': 'default'},
            {'/task/optimizer': 'SGD'},
            '_self_'
        ],
        trainer=dict(
            max_epochs=5
        ),
        task=dict(
            model=dict(
                start_planes=32,
                block=just(PreActBottleneckResNetBlock),
                num_blocks=[3, 3, 3],
                zero_init_residual=False,
                groups=1,
                width_factor=1,
                act_fn=just(nn.SELU),
                norm_layer=just(nn.BatchNorm2d)
            ),
            lr=0.01,
            optimizer=dict(
                momentum=0.9,
                weight_decay=1e-3
            ),
        ),
        datamodule=dict(
            batch_size=128,
            transform=full_builds(transforms.ToTensor)
        ),
        compile_mode=False,
        auto_lr_find=False,
        test=False,
        seed=10,
        bases=(Config,)
    ),
    name='default'
)
