"""This script sweeps the model parameters"""
from typing import Optional

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.basic import test_model


def run(
    *,
    dataset_name: str,
    batch_size: int,
    model_params: ModelParams,
    lr: float,
    count_epoch: int,
) -> float:
    """Run a test"""

    model, loss_fn, test_dataloader, device = train_and_test(
        model_params=model_params,
        lr=lr,
        count_epoch=count_epoch,
        dataset_name=dataset_name,
        batch_size=batch_size,
    )

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)

    return accuracy


def main(
    *,
    model_params_def: Optional[ModelParams] = None,
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
    # pylint:disable=too-many-arguments
    """Main Function"""

    if model_params_def is None:
        model_params_def = ModelParams()

    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        accuracy = run(
            dataset_name=dataset_name,
            batch_size=batch_size,
            model_params=ModelParams(
                conv_out_channels=conv_out_channels,
                kernel_size=kernel_size_def,
                stride=stride,
                padding=padding,
                pool_size=pool_size_def,
            ),
            lr=lr,
            count_epoch=count_epoch,
        )
        print(f"conv_out_channels = {conv_out_channels}: {accuracy*100}%")

    for kernel_size in range(1, 8):
        accuracy = run(
            dataset_name=dataset_name,
            batch_size=batch_size,
            model_params=ModelParams(
                conv_out_channels=conv_out_channels_def,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pool_size=pool_size_def,
            ),
            lr=lr,
            count_epoch=count_epoch,
        )
        print(f"kernel_size = {kernel_size}: {accuracy*100}%")

    for pool_size in range(1, 5):
        accuracy = run(
            dataset_name=dataset_name,
            batch_size=batch_size,
            model_params=ModelParams(
                conv_out_channels=conv_out_channels_def,
                kernel_size=kernel_size_def,
                stride=stride,
                padding=padding,
                pool_size=pool_size,
            ),
            lr=lr,
            count_epoch=count_epoch,
        )
        print(f"pool_size = {pool_size}: {accuracy*100}%")


if __name__ == "__main__":
    main()
