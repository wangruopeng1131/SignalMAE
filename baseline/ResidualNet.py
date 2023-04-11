import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from typing import Tuple, Union, List, Any

from .config import ResNet50, ResNet18


class BottleNeck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_size: List[int],
                 stride: Tuple[int, ...] = 1,
                 down_sampling: bool = False):
        super(BottleNeck, self).__init__()
        stride = 2 if down_sampling else stride

        layer = []
        if len(kernel_size) == 2:

            for k in kernel_size:
                layer.append(
                    nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=k, padding='same'))
                layer.append(nn.BatchNorm1d(out_channels))
                layer.append(nn.GELU())
        elif len(kernel_size) == 3:
            for i, k in enumerate(kernel_size):
                if i < len(kernel_size) - 1:
                    layer.append(
                        nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size[0],
                                  padding='same'))
                    layer.append(nn.BatchNorm1d(hidden_channels))
                    layer.append(nn.GELU())
                else:
                    layer.append(
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size[0],
                                  padding='same'))
                    layer.append(nn.BatchNorm1d(out_channels))
                    layer.append(nn.GELU())

        self.layer = nn.ModuleList(layer)
        self.res_layer = None
        if in_channels != out_channels:
            self.res_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_layer(x) if self.res_layer else x
        residual = x
        for l in self.layer:
            residual = l(residual)
        return x + residual


class ResidualBlock(nn.Module):
    def __init__(self, config: dict):
        super(ResidualBlock, self).__init__()

        layers = [
            BottleNeck(in_channels=items['in_channels'], hidden_channels=items['hidden_channels'],
                       out_channels=items['out_channels'], kernel_size=items['kernel_size'],
                       down_sampling=items['downSampling']) for key, items in config.items()]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResidualNet(nn.Module):
    def __init__(self, config: dict):
        super(ResidualNet, self).__init__()
        self.config = config
        head_config = config['head_config']
        self.resample = Resample(250, 256)
        self.conv = nn.Conv1d(in_channels=head_config['in_channels'], out_channels=head_config['out_channels'],
                              kernel_size=head_config['kernel_size'], stride=head_config['stride'],
                              padding=head_config['padding'])

        self.average = nn.AvgPool1d(kernel_size=head_config['max_pool_kernel'], stride=head_config['max_pool_stride'],
                                    padding=head_config['max_pool_padding'])

        layers = [ResidualBlock(items) for key, items in config['hidden_layers'].items()]
        self.features = nn.ModuleList(layers)

        self.average_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=config['out_channels'], out_features=config['classes'])

    def forward(self, x: torch.Tensor, input_second: int) -> torch.Tensor:
        B, C, L = x.shape  # Batch, 1, 16 * 250
        x_resample = self.resample(x)  # noramlize 1, 16 * 250
        x = F.layer_norm(x, [C, L])
        x_resample = self.conv(x_resample)  # Batch, 1, 16 * 250 -> Batch, 64, 16 * 125
        x_resample = self.average(x_resample)
        for layer in self.features:
            x_resample = layer(x_resample)
        x_resample = torch.torch.einsum('bcs->bsc', x_resample)
        B, L, C = x_resample.shape
        x_resample = x_resample.reshape(B * input_second, L // input_second, -1)
        x_resample = torch.torch.einsum('bcs->bsc', x_resample)
        x_resample = self.average_pool(x_resample)
        x_resample = x_resample.view(B * input_second, -1)
        x_resample = self.classifier(x_resample)
        return x_resample


def load_model(name):
    if name == 'resnet18':
        model = ResidualNet(ResNet18)
        return model
    elif name == 'resnet50':
        model = ResidualNet(ResNet50)
        return model
