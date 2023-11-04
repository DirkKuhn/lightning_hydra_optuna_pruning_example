from hydra_zen import zen
import pytorch_lightning as pl

from run import train_nn

# Use `zen` to automatically instantiate the configured objects
zen_run = zen(train_nn, pre_call=lambda cfg: pl.seed_everything(cfg.seed, workers=True))


if __name__ == '__main__':
    # Execute configuration files, which add configurations to the hydra store
    import config_hydra_zen  # noqa
    # Add command-line interface
    zen_run.hydra_main(
        config_name='train_nn',
        config_path=None,
        version_base='1.3'
    )
