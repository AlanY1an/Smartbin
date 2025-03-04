# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/25 15:48
# @Author : 林枫
# @File : DenseNet.py

import re
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from torchvision.models import WeightsEnum
from torchvision.utils import _log_api_usage_once
from model.config import NUM_CLASSES



__all__ = [
    "DenseNet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]
# official pretrain weights
model_urls = {
    'DenseNet121_Weights': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'DenseNet161_Weights': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'DenseNet169_Weights': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'DenseNet201_Weights': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
}


class _DenseLayer(nn.Module):
    def __init__(
            self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float,
            memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input, use_reentrant=False)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
            self,
            num_layers: int,
            num_input_features: int,
            bn_size: int,
            growth_rate: int,
            drop_rate: float,
            memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    def __init__(
            self,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress, check_hash=True)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
        growth_rate: int,
        block_config: Tuple[int, int, int, int],
        num_init_features: int,
        **kwargs: Any,
) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    return model


def densenet121(**kwargs: Any) -> DenseNet:
    return _densenet(32, (6, 12, 24, 16), 64, **kwargs)


def densenet161(**kwargs: Any) -> DenseNet:
    return _densenet(48, (6, 12, 36, 24), 96, **kwargs)


def densenet169(**kwargs: Any) -> DenseNet:
    return _densenet(32, (6, 12, 32, 32), 64, **kwargs)


def densenet201(**kwargs: Any) -> DenseNet:
    return _densenet(32, (6, 12, 48, 32), 64, **kwargs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())

    # Check CUDA version
    cuda_version = torch.version.cuda
    print("cuDA Version:", cuda_version)

    # Check CuDNN version

    cudnn_version = torch.backends.cudnn.version()
    print("CuDNN Version:", cudnn_version)
    model = densenet121(num_classes=NUM_CLASSES).to(device)
    print(model)
    # print(summary(model, (3, 227, 227)))