from collections.abc import Callable
from typing import Optional, Type, Union

import torch as th
from torch import nn
from torch.nn.init import calculate_gain


def _calculate_gain(nonlinearity: Optional[nn.Module], param=None, *args, **kwargs) -> float:
    if isinstance(nonlinearity, None | nn.Sigmoid | nn.Softmax):
        return 1
    elif isinstance(nonlinearity, nn.ReLU | nn.ELU | nn.PReLU | nn.RReLU | nn.Mish):
        return 2 ** 0.5
    elif isinstance(nonlinearity, nn.SELU):
        return 1  # alt: 3.0/4, 1 for self-normalizing neural networks
    elif isinstance(nonlinearity, nn.Tanh):
        return 5.0 / 3  # alt: 1
    elif isinstance(nonlinearity, nn.LeakyReLU):
        return (2.0 / (1 + nonlinearity.negative_slope ** 2)) ** 0.5
    else:
        try:
            # Default cases on which pytorch relies
            return calculate_gain(nonlinearity, param)
        except:
            raise ValueError(
                f'Unexpected non-linearity {nonlinearity}, '
                f'please add default values.'
            )


# Use custom gains
nn.init.calculate_gain = _calculate_gain


class ResNet(nn.Module):
    def __init__(
            self,
            start_planes: int,
            block: Type[Union['PreActResNetBlock', 'PreActBottleneckResNetBlock']],
            num_blocks: list[int],
            num_classes: int,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_factor: float = 1,
            act_fn: Callable[[], nn.Module] = nn.ReLU,
            norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d
    ):
        super().__init__()

        self.in_planes = start_planes

        self.conv1 = nn.Conv2d(
            1, start_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.layers = nn.Sequential(*[
            self._make_layer(
                block=block, planes=start_planes * 2**i, num_blocks=nb,
                groups=groups, width_factor=width_factor,
                act_fn=act_fn, norm_layer=norm_layer
            )
            for i, nb in enumerate(num_blocks)
        ])
        out_planes = start_planes * 2**(len(num_blocks)-1) * block.expansion
        self.norml = norm_layer(out_planes)
        self.actl = act_fn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity=act_fn()
                )

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PreActBottleneckResNetBlock):
                    nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, PreActResNetBlock):
                    nn.init.constant_(m.norm2.weight, 0)

    def _make_layer(
            self,
            block: Type[Union['PreActResNetBlock', 'PreActBottleneckResNetBlock']],
            planes: int,
            num_blocks: int,
            groups: int,
            width_factor: float,
            act_fn: Callable[[], nn.Module],
            norm_layer: Callable[[int], nn.Module]
    ) -> nn.Sequential:
        stride = 2
        blocks = [
            block(
                self.in_planes, planes, stride=stride, act_fn=act_fn,
                norm_layer=norm_layer, groups=groups, width_factor=width_factor
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            blocks.append(
                block(
                    self.in_planes, planes, stride=1, act_fn=act_fn,
                    norm_layer=norm_layer, groups=groups, width_factor=width_factor

                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv1(x)
        x = self.layers(x)
        x = self.norml(x)
        x = self.actl(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        return x


class PreActResNetBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            act_fn: Callable[[], nn.Module],
            norm_layer: Callable[[int], nn.Module],
            groups: int,
            width_factor: float
    ):
        super().__init__()

        if groups != 1 or width_factor != 1:
            raise ValueError("BasicBlock only supports groups=1 and width_factor=1")

        self.norm1 = norm_layer(in_planes)
        self.act1 = act_fn()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = norm_layer(planes)
        self.act2 = act_fn()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.downsample = create_preact_downsample(
            in_planes, planes, stride=stride, expansion=self.expansion, norm_layer=norm_layer
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        z = self.norm1(x)
        z = self.act1(z)
        z = self.conv1(z)

        z = self.norm2(z)
        z = self.act2(z)
        z = self.conv2(z)

        if self.downsample:
            x = self.downsample(x)

        z += x

        return z


class PreActBottleneckResNetBlock(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int,
            act_fn: Callable[[], nn.Module],
            norm_layer: Callable[[int], nn.Module],
            groups: int,
            width_factor: float
    ):
        super().__init__()
        width = int(planes * width_factor) * groups

        self.norm1 = norm_layer(in_planes)
        self.act1 = act_fn()
        self.conv1 = nn.Conv2d(
            in_planes, width, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.norm2 = norm_layer(width)
        self.act2 = act_fn()
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.norm3 = norm_layer(width)
        self.act3 = act_fn()
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.downsample = create_preact_downsample(
            in_planes, planes, stride=stride, expansion=self.expansion, norm_layer=norm_layer
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        z = self.norm1(x)
        z = self.act1(z)
        z = self.conv1(z)

        z = self.norm2(z)
        z = self.act2(z)
        z = self.conv2(z)

        z = self.norm3(z)
        z = self.act3(z)
        z = self.conv3(z)

        if self.downsample:
            x = self.downsample(x)

        z += x

        return z


def create_preact_downsample(
        in_planes: int,
        planes: int,
        stride: int,
        expansion: int,
        norm_layer: Callable[[int], nn.Module]
) -> Optional[nn.Sequential]:
    downsample = None
    if stride != 1 or in_planes != planes * expansion:
        downsample = nn.Sequential(
            norm_layer(in_planes),
            nn.Conv2d(
                in_planes, planes * expansion, kernel_size=1, stride=stride, padding=0, bias=False
            )
        )
    return downsample
