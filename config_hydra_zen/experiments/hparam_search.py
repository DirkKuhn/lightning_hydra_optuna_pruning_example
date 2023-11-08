from hydra_zen import just, make_config
from hydra_plugins.hydra_optuna_pruning_sweeper import ListSearchSpace
from torch import nn
from torchvision import transforms

from ..config import Config
from ..utils import full_builds, experiment_store
from ..model import PreActBottleneckResNetBlock


experiment_store(
    make_config(
        hydra_defaults=[
            {'/trainer/callbacks': 'pruning'},
            {'/task/optimizer': 'SGD'},
            {'/hydra/sweeper/sampler': 'TPESampler'},
            {'/hydra/sweeper/pruner': 'HyperbandPruner'},
            {'override /hydra/sweeper': 'OptunaPruningSweeper'},
            '_self_'
        ],
        trainer=dict(
            max_epochs=10
        ),
        task=dict(
            model=dict(
                block=just(PreActBottleneckResNetBlock),
                groups=1,
                width_factor=1,
                norm_layer=just(nn.BatchNorm2d)
            ),
        ),
        datamodule=dict(
            transform=full_builds(transforms.ToTensor)
        ),
        compile_mode=False,
        auto_lr_find=False,
        test=False,
        seed=10,
        hydra=dict(
            sweep=dict(
                dir='optuna/${hydra.sweeper.study_name}'
            ),
            sweeper=dict(
                sampler=dict(
                    seed=123
                ),
                direction='maximize',
                storage='sqlite:///${.study_name}.db',
                study_name='resnet',
                n_trials=100,
                n_jobs=1,
                params={
                    '+task.model.start_planes': 'tag("16", range(8, 64))',
                    'task.model.zero_init_residual': 'tag("false", choice(false, true))',
                    '+task/model/act_fn': 'tag(ReLU, choice(ReLU, LeakyReLU, SELU, Mish))',
                    '+task.lr': 'tag("0.01", log, interval(1e-4, 0.05))',
                    '+optimizer.momentum': 'tag("0.9", interval(0.5, 1))',
                    '+optimizer.weight_decay': 'tag("1e-3", log, interval(1e-5, 1e-2))',
                    '+datamodule.batch_size': 'tag("128", range(64, 256))'
                },
                custom_search_space=full_builds(
                    ListSearchSpace,
                    name='task.model.num_blocks',
                    min_entries=2,
                    max_entries=4,
                    min_value=1,
                    max_value=4,
                    use_float=False,
                    manual_values=[[3, 3, 3]]
                ),
                gc_after_trial=True

            )
        ),
        bases=(Config,)
    ),
    name='hparam_search'
)
