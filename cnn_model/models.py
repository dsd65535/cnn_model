"""This module defines pyTorch modules and layers"""
import torch


class Ideal(torch.nn.Module):
    """An ideal version of the CNN architecture presented in:

    Nikita Mirchandani. Ultra-Low Power and Robust Analog Computing
    Circuits and System Design Framework for Machine Learning Applications.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(4608, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Total forward computation"""

        return self.layers(x)
