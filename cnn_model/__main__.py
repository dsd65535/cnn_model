"""This script demonstrates high-level functionality"""
from typing import Tuple

import torch
import torchvision

from cnn_model.basic import extract_shape
from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.models import Ideal

DATACACHEDIR = "cache/data"


def get_mnist(
    batch_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get MNIST dataset"""

    training_data = torchvision.datasets.MNIST(
        root=DATACACHEDIR,
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_data = torchvision.datasets.MNIST(
        root=DATACACHEDIR,
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def get_input_parameters(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
) -> Tuple[int, int, int]:
    """Get the parameters of the input"""

    train_shape, train_label_count = extract_shape(train_dataloader)
    test_shape, test_label_count = extract_shape(test_dataloader)

    if train_shape != test_shape:
        raise RuntimeError("Mismatched test shape")

    if train_label_count != test_label_count:
        raise RuntimeError("Mismatched test label count")

    if len(train_shape) != 3:
        raise RuntimeError("Bad input shape")

    if train_shape[2] != train_shape[1]:
        raise RuntimeError("Non-square input")

    return train_shape[0], train_shape[1], train_label_count


def main(
    *,
    lr: float = 1e-3,
    count_epoch: int = 5,
    batch_size: int = 1,
    print_rate: int = 1000,
    conv_out_channels: int = 32,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    pool_size: int = 2,
) -> None:
    # pylint:disable=too-many-arguments,too-many-locals
    """Train and Test the Ideal model

    This function is based on:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    device = get_device()

    train_dataloader, test_dataloader = get_mnist(batch_size=batch_size)
    in_channels, in_size, feature_count = get_input_parameters(
        train_dataloader, test_dataloader
    )

    model = Ideal(
        in_size=in_size,
        in_channels=in_channels,
        conv_out_channels=conv_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        feature_count=feature_count,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for idx_epoch in range(count_epoch):
        print(f"Epoch {idx_epoch+1}/{count_epoch}:")
        train_model(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device=device,
            print_rate=print_rate,
        )
        avg_loss, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
        print(f"Average Loss:  {avg_loss:<9f}")
        print(f"Accuracy:      {(100*accuracy):<0.4f}%")
        print()


if __name__ == "__main__":
    main()
