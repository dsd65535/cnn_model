"""This script trains and tests the Ideal model on the MNIST dataset"""
import argparse
from pathlib import Path
from typing import Optional
from typing import Tuple

import torch
import torchvision

from cnn_model.basic import extract_shape
from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.common import DATACACHEDIR
from cnn_model.common import MODELCACHEDIR
from cnn_model.models import Ideal


def get_mnist(
    batch_size: int = 1,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Get MNIST dataset"""

    DATACACHEDIR.mkdir(parents=True, exist_ok=True)

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
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    cache_filepath = Path(
        f"{MODELCACHEDIR}/ideal_mnist_"
        f"{lr}_{count_epoch}_{batch_size}_"
        f"{conv_out_channels}_{kernel_size}_{stride}_{padding}_{pool_size}.pth"
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
    parser.add_argument("--print_rate", type=int, nargs="?")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--retrain", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Main Function"""

    args = parse_args()

    train_and_test_ideal_mnist(
        lr=args.lr,
        count_epoch=args.count_epoch,
        batch_size=args.batch_size,
        conv_out_channels=args.conv_out_channels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        pool_size=args.pool_size,
        print_rate=args.print_rate,
        use_cache=not args.no_cache,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
