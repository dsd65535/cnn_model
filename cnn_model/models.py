"""This module defines pyTorch modules and layers"""
import math
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Tuple

import torch


@dataclass
class FullModelParams:
    # pylint:disable=too-many-instance-attributes  # dataclass
    """Parameters for Main model"""

    in_size: int
    in_channels: int
    conv_out_channels: int
    kernel_size: int
    stride: int
    padding: int
    pool_size: int
    feature_count: int
    additional_layers: Optional[List[int]]

    conv_out_size: int = field(init=False)
    pool_out_size: int = field(init=False)
    final_size: int = field(init=False)
    additional_layer_sizes: List[Tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.conv_out_size, rem = divmod(
            self.in_size + self.stride + 2 * self.padding - self.kernel_size,
            self.stride,
        )
        if rem:
            raise ValueError("Invalid Convolution Output Size")

        self.pool_out_size, rem = divmod(self.conv_out_size, self.pool_size)
        if rem:
            raise ValueError("Invalid Pool Output Size")

        current_size = self.pool_out_size**2 * self.conv_out_channels
        additional_layer_sizes = []
        if self.additional_layers is not None:
            for next_size in self.additional_layers:
                additional_layer_sizes.append((current_size, next_size))
                current_size = next_size

        self.final_size = current_size
        self.additional_layer_sizes = additional_layer_sizes


@dataclass
class Nonidealities:
    """Nonidealities parameters for Main model"""

    relu_cutoff: float = 0.0
    relu_out_noise: Optional[float] = None
    linear_out_noise: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.relu_cutoff}_{self.relu_out_noise}_{self.linear_out_noise}"


@dataclass
class Normalization:
    """Normalization parameters for Main model"""

    min_out: float = 0.0
    max_out: float = 1.0
    min_in: float = 0.0
    max_in: float = 1.0


class Normalize(torch.nn.Module):
    """Layer that converts to a realistic voltage"""

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
    """A configurable CNN model similar to the one presented in:

    Nikita Mirchandani. Ultra-Low Power and Robust Analog Computing
    Circuits and System Design Framework for Machine Learning Applications.
    """

    def __init__(
        self,
        full_model_params: FullModelParams,
        nonidealities: Optional[Nonidealities] = None,
        normalization: Optional[Normalization] = None,
    ) -> None:
        super().__init__()

        if nonidealities is None:
            nonidealities = Nonidealities()

        layers: List[torch.nn.Module] = []

        if normalization is not None:
            layers.append(
                Normalize(
                    normalization.min_out,
                    normalization.max_out,
                    normalization.min_in,
                    normalization.max_in,
                )
            )
            layers.append(
            torch.nn.Conv2d(
                full_model_params.in_channels,
                full_model_params.conv_out_channels,
                full_model_params.kernel_size,
                full_model_params.stride,
                full_model_params.padding,
            )
        )
            layers.append(
            torch.nn.Conv2d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=(3,3), 
            padding=(1,1)))

        layers.append(ReLU(nonidealities.relu_cutoff, nonidealities.relu_out_noise))
        layers.append(torch.nn.MaxPool2d(full_model_params.pool_size))
        layers.append(
            torch.nn.Conv2d(
            in_channels=64, 
            out_channels=96, 
            kernel_size=(3,3), 
            padding=(1,1)))
        layers.append(torch.nn.MaxPool2d(full_model_params.pool_size))

        layers.append(torch.nn.Flatten())
        # softmax, drop out, dense layers
        for in_size, out_size in full_model_params.additional_layer_sizes:
            layers.append(Linear(in_size, out_size))
            layers.append(ReLU())

        layers.append(
            Linear(
                4704,
                full_model_params.feature_count,
                nonidealities.linear_out_noise,
            )
        )

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Total forward computation"""

        return self.layers(x)
