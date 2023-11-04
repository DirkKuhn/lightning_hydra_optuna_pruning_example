from collections.abc import Callable, Iterable

import torch as th
from torch import nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy


class ClassificationTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            lr: float,
            optimizer: Callable[[Iterable[th.Tensor], float], optim.Optimizer],
            num_classes: int,
            input_dim: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.opt = optimizer
        self._init_metrics(num_classes)
        self.example_input_array = th.zeros(1, *input_dim)

    def _init_metrics(self, num_classes: int) -> None:
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = self.train_acc.clone()
        self.test_acc = self.train_acc.clone()

    def forward(self, data: th.Tensor) -> th.Tensor:
        return self.model(data)

    def training_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> th.Tensor:
        output = self._helper(batch, "train")
        _, target = batch
        loss = F.cross_entropy(output, target)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> None:
        self._helper(batch, "val")

    def test_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int) -> None:
        self._helper(batch, "test")

    def _helper(self, batch: tuple[th.Tensor, th.Tensor], stage: str) -> th.Tensor:
        data, target = batch
        output = self(data)
        acc: Metric = getattr(self, f"{stage}_acc")
        acc(output, target)
        self.log(f"{stage}/{acc.__class__.__name__}", acc, prog_bar=True)
        return output

    def configure_optimizers(self) -> optim.Optimizer:
        return self.opt(self.parameters(), lr=self.lr)
