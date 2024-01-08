"""This module defines pyTorch modules and layers"""
import torch


class Ideal(torch.nn.Module):
    """An ideal version of the CNN architecture presented in:

    Nikita Mirchandani. Ultra-Low Power and Robust Analog Computing
    Circuits and System Design Framework for Machine Learning Applications.
    """

    def __init__(
        self,
        *,
        in_size: int = 28,
        in_channels: int = 1,
        conv_out_channels: int = 32,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
        feature_count: int = 10,
    ) -> None:
        # pylint:disable=too-many-arguments

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
            torch.nn.Conv2d(
                in_channels, conv_out_channels, kernel_size, stride, padding
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(pool_size),
            torch.nn.Flatten(),
            torch.nn.Linear(flattened_size, feature_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Total forward computation"""

        return self.layers(x)
