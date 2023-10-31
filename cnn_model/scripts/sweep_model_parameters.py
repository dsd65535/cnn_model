# pylint:disable=duplicate-code
"""This script sweeps the model parameters"""
import argparse
import logging
import time
from dataclasses import replace
from typing import Optional

import git

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.__main__ import TrainParams
from cnn_model.basic import test_model
from cnn_model.models import Nonidealities
from cnn_model.models import Normalization
from cnn_model.parser import add_arguments_from_dataclass_fields


def run(
    *,
    dataset_name: str,
    train_params: Optional[TrainParams],
    model_params: ModelParams,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    use_cache: bool = True,
    retrain: bool = False,
    print_rate: Optional[int] = None,
) -> Optional[float]:
    # pylint:disable=too-many-arguments
    """Run a test"""

    try:
        model, loss_fn, test_dataloader, device = train_and_test(
            dataset_name=dataset_name,
            train_params=train_params,
            model_params=model_params,
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=use_cache,
            retrain=retrain,
            print_rate=print_rate,
        )
    except ValueError:
        logging.exception("Training failed")
        return None

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)

    return accuracy


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_name", type=str, default="MNIST")
    add_arguments_from_dataclass_fields(TrainParams, parser)
    add_arguments_from_dataclass_fields(ModelParams, parser)
    add_arguments_from_dataclass_fields(Nonidealities, parser)
    add_arguments_from_dataclass_fields(Normalization, parser)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--print_rate", type=int, nargs="?")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

    return parser.parse_args()


def main() -> None:
    # pylint:disable=too-many-branches,too-many-locals
    """Main Function"""

    args = parse_args()

    if args.print_git_info:
        repo = git.Repo(search_parent_directories=True)
        print(f"Git SHA: {repo.head.object.hexsha}")
        diff = repo.git.diff()
        if diff:
            print(repo.git.diff())
        print()

    if args.timed:
        start = time.time()

    train_params = TrainParams(
        count_epoch=args.count_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        noise_train=args.noise_train,
    )
    model_params_def = ModelParams(
        conv_out_channels=args.conv_out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        pool_size=args.pool_size,
        additional_layers=args.additional_layers,
    )
    nonidealities = Nonidealities(
        relu_cutoff=args.relu_cutoff,
        relu_out_noise=args.relu_out_noise,
        linear_out_noise=args.linear_out_noise,
    )
    normalization = Normalization(
        min_out=args.min_out,
        max_out=args.max_out,
        min_in=args.min_in,
        max_in=args.max_in,
    )

    for conv_out_channels_exp in range(8):
        conv_out_channels = 2**conv_out_channels_exp
        accuracy = run(
            dataset_name=args.dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, conv_out_channels=conv_out_channels),
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=not args.no_cache,
            retrain=args.retrain,
            print_rate=args.print_rate,
        )
        if accuracy is None:
            print(f"conv_out_channels = {conv_out_channels}: N/A")
        else:
            print(f"conv_out_channels = {conv_out_channels}: {accuracy*100}%")

    for kernel_size in range(1, 8):
        accuracy = run(
            dataset_name=args.dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, kernel_size=kernel_size),
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=not args.no_cache,
            retrain=args.retrain,
            print_rate=args.print_rate,
        )
        if accuracy is None:
            print(f"kernel_size = {kernel_size}: N/A")
        else:
            print(f"kernel_size = {kernel_size}: {accuracy*100}%")

    for pool_size in range(1, 5):
        accuracy = run(
            dataset_name=args.dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, pool_size=pool_size),
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=not args.no_cache,
            retrain=args.retrain,
            print_rate=args.print_rate,
        )
        if accuracy is None:
            print(f"pool_size = {pool_size}: N/A")
        else:
            print(f"pool_size = {pool_size}: {accuracy*100}%")

    for additional_layer_exp in range(5, 10):
        additional_layers = [2**additional_layer_exp]
        accuracy = run(
            dataset_name=args.dataset_name,
            train_params=train_params,
            model_params=replace(model_params_def, additional_layers=additional_layers),
            nonidealities=nonidealities,
            normalization=normalization,
            use_cache=not args.no_cache,
            retrain=args.retrain,
            print_rate=args.print_rate,
        )
        if accuracy is None:
            print(f"additional_layers = {additional_layers}: N/A")
        else:
            print(f"additional_layers = {additional_layers}: {accuracy*100}%")

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
