# pylint:disable=duplicate-code
"""This script sweeps the model parameters"""
import torch

from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.datasets import get_dataset
from cnn_model.datasets import get_input_parameters
from cnn_model.models import Main


def run(
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    *,
    device: str,
    lr: float,
    count_epoch: int,
    in_size: int,
    in_channels: int,
    conv_out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    pool_size: int,
    feature_count: int,
) -> float:
    # pylint:disable=too-many-arguments,too-many-locals
    """Run a test"""

    model = Main(
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

    for _ in range(count_epoch):
        train_model(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device=device,
        )

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)

    return accuracy


def main(
    *,
    lr: float = 1e-3,
    count_epoch: int = 5,
    dataset_name: str = "MNIST",
    batch_size: int = 1,
    conv_out_channels_def: int = 32,
    kernel_size_def: int = 5,
    stride: int = 1,
    padding: int = 0,
    pool_size_def: int = 2,
) -> None:
    # pylint:disable=too-many-arguments,too-many-locals
    """Main Function"""

    device = get_device()

    train_dataloader, test_dataloader = get_dataset(
        name=dataset_name, batch_size=batch_size
    )
    in_channels, in_size, feature_count = get_input_parameters(
        train_dataloader, test_dataloader
    )

    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        accuracy = run(
            train_dataloader,
            test_dataloader,
            device=device,
            lr=lr,
            count_epoch=count_epoch,
            in_size=in_size,
            in_channels=in_channels,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size_def,
            stride=stride,
            padding=padding,
            pool_size=pool_size_def,
            feature_count=feature_count,
        )
        print(f"conv_out_channels = {conv_out_channels}: {accuracy*100}%")

    for kernel_size in range(1, 8):
        accuracy = run(
            train_dataloader,
            test_dataloader,
            device=device,
            lr=lr,
            count_epoch=count_epoch,
            in_size=in_size,
            in_channels=in_channels,
            conv_out_channels=conv_out_channels_def,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_size=pool_size_def,
            feature_count=feature_count,
        )
        print(f"kernel_size = {kernel_size}: {accuracy*100}%")

    for pool_size in range(1, 5):
        accuracy = run(
            train_dataloader,
            test_dataloader,
            device=device,
            lr=lr,
            count_epoch=count_epoch,
            in_size=in_size,
            in_channels=in_channels,
            conv_out_channels=conv_out_channels_def,
            kernel_size=kernel_size_def,
            stride=stride,
            padding=padding,
            pool_size=pool_size,
            feature_count=feature_count,
        )
        print(f"pool_size = {pool_size}: {accuracy*100}%")


if __name__ == "__main__":
    main()
