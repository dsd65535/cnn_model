"""This script sweeps the model parameters"""
import logging
from dataclasses import replace
from typing import Optional

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.__main__ import TrainParams
from cnn_model.basic import test_model


def run(
    *,
    dataset_name: str,
    train_params: Optional[TrainParams],
    model_params: ModelParams,
) -> Optional[float]:
    """Run a test"""

    try:
        model, loss_fn, test_dataloader, device = train_and_test(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=model_params,
        )
    except ValueError:
        logging.exception("Training failed")
        return None

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)

    return accuracy


def main(
    *,
    dataset_name: str = "MNIST",
    train_params: Optional[TrainParams] = None,
    model_params_def: Optional[ModelParams] = None,
) -> None:
    """Main Function"""

    if model_params_def is None:
        model_params_def = ModelParams()

    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        accuracy = run(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, conv_out_channels=conv_out_channels),
        )
        if accuracy is None:
            print(f"conv_out_channels = {conv_out_channels}: N/A")
        else:
            print(f"conv_out_channels = {conv_out_channels}: {accuracy*100}%")

    for kernel_size in range(1, 8):
        accuracy = run(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, kernel_size=kernel_size),
        )
        if accuracy is None:
            print(f"kernel_size = {kernel_size}: N/A")
        else:
            print(f"kernel_size = {kernel_size}: {accuracy*100}%")

    for pool_size in range(1, 5):
        accuracy = run(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, pool_size=pool_size),
        )
        if accuracy is None:
            print(f"pool_size = {pool_size}: N/A")
        else:
            print(f"pool_size = {pool_size}: {accuracy*100}%")


if __name__ == "__main__":
    main()
