"""This script trains and tests the Ideal model on the MNIST dataset"""
import argparse
import time
from pathlib import Path
from typing import Optional
from typing import Tuple

import git
import torch

from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.common import MODELCACHEDIR
from cnn_model.datasets import get_input_parameters
from cnn_model.datasets import get_mnist
from cnn_model.models import Ideal


def train_and_test_ideal_mnist(
    *,
    lr: float = 1e-3,
    count_epoch: int = 5,
    batch_size: int = 1,
    conv_out_channels: int = 32,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    pool_size: int = 2,
    relu_cutoff: float = 0.0,
    relu_out_noise: Optional[float] = None,
    linear_out_noise: Optional[float] = None,
    print_rate: Optional[int] = None,
    use_cache: bool = True,
    retrain: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.utils.data.DataLoader, str]:
    # pylint:disable=too-many-arguments,too-many-locals
    """Train and Test the Ideal model on MNIST

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
        relu_cutoff=relu_cutoff,
        relu_out_noise=relu_out_noise,
        linear_out_noise=linear_out_noise,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    cache_filepath = Path(
        f"{MODELCACHEDIR}/ideal_mnist_"
        f"{lr}_{count_epoch}_{batch_size}_"
        f"{conv_out_channels}_{kernel_size}_{stride}_{padding}_{pool_size}"
        f"_{relu_cutoff}_{relu_out_noise}_{linear_out_noise}.pth"
    )

    if not use_cache or retrain or not cache_filepath.exists():
        for idx_epoch in range(count_epoch):
            if print_rate is not None:
                print(f"Epoch {idx_epoch+1}/{count_epoch}:")
            train_model(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
                print_rate=print_rate,
            )
            if print_rate is not None and idx_epoch < count_epoch - 1:
                avg_loss, accuracy = test_model(
                    model, test_dataloader, loss_fn, device=device
                )
                print(f"Average Loss:  {avg_loss:<9f}")
                print(f"Accuracy:      {(100*accuracy):<0.4f}%")
                print()

        if use_cache:
            MODELCACHEDIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cache_filepath)
    else:
        model.load_state_dict(torch.load(cache_filepath))

    if print_rate is not None:
        avg_loss, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
        print(f"Average Loss:  {avg_loss:<9f}")
        print(f"Accuracy:      {(100*accuracy):<0.4f}%")

    return model, loss_fn, test_dataloader, device


def parse_args() -> argparse.Namespace:
    """Parse CLI Arguments"""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--count_epoch", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--conv_out_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--relu_cutoff", type=float, default=0.0)
    parser.add_argument("--relu_out_noise", type=float, nargs="?")
    parser.add_argument("--linear_out_noise", type=float, nargs="?")
    parser.add_argument("--print_rate", type=int, nargs="?")
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

    train_and_test_ideal_mnist(
        lr=args.lr,
        count_epoch=args.count_epoch,
        batch_size=args.batch_size,
        conv_out_channels=args.conv_out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        pool_size=args.pool_size,
        relu_cutoff=args.relu_cutoff,
        relu_out_noise=args.relu_out_noise,
        linear_out_noise=args.linear_out_noise,
        print_rate=args.print_rate,
        use_cache=not args.no_cache,
        retrain=args.retrain,
    )

    if args.timed:
        end = time.time()
        print(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
