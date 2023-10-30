# pylint:disable=R0801
"""This script determines the effect of noise"""
import argparse
import json
import time
from pathlib import Path
from typing import List
from typing import Optional

import git

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.basic import test_model
from cnn_model.models import Nonidealities


def run(
    noises_train: List[Optional[float]],
    noises_test: List[Optional[float]],
    output_filepath: Path,
    *,
    lr: float = 1e-3,
    count_epoch: int = 5,
    dataset_name: str = "MNIST",
    batch_size: int = 1,
    conv_out_channels: int = 32,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    pool_size: int = 2,
    relu_cutoff: float = 0.0,
    relu_out_noise: Optional[float] = None,
    linear_out_noise: Optional[float] = None,
    use_cache: bool = True,
    retrain: bool = False,
) -> None:
    # pylint:disable=too-many-arguments,too-many-locals
    """Run"""

    full_results = {}
    for noise_train in noises_train:
        results = {}

        model, loss_fn, test_dataloader, device = train_and_test(
            model_params=ModelParams(
                conv_out_channels=conv_out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                pool_size=pool_size,
            ),
            nonidealities=Nonidealities(
                relu_cutoff=relu_cutoff,
                relu_out_noise=relu_out_noise,
                linear_out_noise=linear_out_noise,
            ),
            lr=lr,
            count_epoch=count_epoch,
            dataset_name=dataset_name,
            batch_size=batch_size,
            noise_train=noise_train,
            use_cache=use_cache,
            retrain=retrain,
        )

        for noise_test in noises_test:
            result = test_model(
                model, test_dataloader, loss_fn, device=device, noise=noise_test
            )
            print(f"{noise_train} {noise_test} {result}")
            results[noise_test] = result

        full_results[noise_train] = results

    with output_filepath.open("w") as output_file:
        json.dump(full_results, output_file, indent=4)


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("output_filepath", type=Path)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--count_epoch", type=int, default=5)
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--conv_out_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--relu_cutoff", type=float, default=0.0)
    parser.add_argument("--relu_out_noise", type=float, nargs="?")
    parser.add_argument("--linear_out_noise", type=float, nargs="?")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--timed", action="store_true")
    parser.add_argument("--print_git_info", action="store_true")

    return parser.parse_args()


def main() -> None:
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

    run(
        [None] + [2**exp for exp in range(-4, 4)],
        [None] + [2 ** (exp / 10) for exp in range(-40, 40)],
        args.output_filepath,
        lr=args.lr,
        count_epoch=args.count_epoch,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        conv_out_channels=args.conv_out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        pool_size=args.pool_size,
        relu_cutoff=args.relu_cutoff,
        relu_out_noise=args.relu_out_noise,
        linear_out_noise=args.linear_out_noise,
        use_cache=not args.no_cache,
        retrain=args.retrain,
    )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
