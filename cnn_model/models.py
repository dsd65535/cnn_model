"""This module defines pyTorch modules and layers"""
import math
from typing import Optional

import torch


class Normalize(torch.nn.Module):
    """Layer that convers to a realistic voltage"""

    def __init__(
        self, min_out: float, max_out: float, min_in: float = 0.0, max_in: float = 1.0
    ) -> None:
        super().__init__()

        self.slope = (max_out - min_out) / (max_in - min_in)
        self.offset = min_out - self.slope * min_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        return torch.add(self.offset, torch.multiply(self.slope, x))


class ReLU(torch.nn.Module):
    """Re-implementation of ReLU"""

    def __init__(self, cutoff: float = 0.0, out_noise: Optional[float] = None) -> None:
        super().__init__()

        self.cutoff = torch.tensor([cutoff])

        self.out_noise = out_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        out = torch.max(self.cutoff.to(x.device), x)
        if self.out_noise is not None:
            out = torch.add(out, self.out_noise * torch.randn(out.shape).to(out.device))

        return out


class Linear(torch.nn.Module):
    """Re-implementation of Linear"""

    def __init__(
        self, in_features: int, out_features: int, out_noise: Optional[float] = None
    ) -> None:
        super().__init__()

        self.in_features, self.out_features = in_features, out_features

        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))

        torch.nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.out_noise = out_noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function"""

        out = torch.add(torch.mm(x, self.weight.t()), self.bias)
        if self.out_noise is not None:
            out = torch.add(out, self.out_noise * torch.randn(out.shape).to(out.device))

        return out


class Main(torch.nn.Module):
    """A CNN architecture similar to the one presented in:

    Nikita Mirchandani. Ultra-Low Power and Robust Analog Computing
    Circuits and System Design Framework for Machine Learning Applications.
    """

    def __init__(
        self,
        *,
        min_out: float = 0.0,
        max_out: float = 1.0,
        min_in: float = 0.0,
        max_in: float = 1.0,
        in_size: int = 28,
        in_channels: int = 1,
        conv_out_channels: int = 32,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
        feature_count: int = 10,
        relu_cutoff: float = 0.5,
        relu_out_noise: Optional[float] = None,
        linear_out_noise: Optional[float] = None,
    ) -> None:
        # pylint:disable=too-many-arguments,too-many-locals

        super().__init__()

        conv_out_size, rem = divmod(
            in_size + stride + 2 * padding - kernel_size, stride
        )
        if rem:
            raise ValueError("Invalid Convolution Output Size")

        pool_out_size, rem = divmod(conv_out_size, pool_size)
        if rem:
            raise ValueError("Invalid Pool Output Size")

        flattened_size = pool_out_size**2 * conv_out_channels

        self.layers = torch.nn.Sequential(
            Normalize(min_out, max_out, min_in, max_in),
            torch.nn.Conv2d(
                in_channels, conv_out_channels, kernel_size, stride, padding
            ),
            ReLU(relu_cutoff, relu_out_noise),
            torch.nn.MaxPool2d(pool_size),
            torch.nn.Flatten(),
            Linear(flattened_size, feature_count, linear_out_noise),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Total forward computation"""

        return self.layers(x)
