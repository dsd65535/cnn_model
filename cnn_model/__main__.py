# pylint:disable=logging-fstring-interpolation
"""This script trains and tests the Main model"""
import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import git
import torch

from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.datasets import get_dataset_and_params
from cnn_model.models import FullModelParams
from cnn_model.models import Main
from cnn_model.models import Nonidealities
from cnn_model.models import Normalization
from cnn_model.normalize import normalize_values
from cnn_model.parser import add_arguments_from_dataclass_fields

MODELCACHEDIR = Path("cache/models")


@dataclass
class TrainParams:
    """Parameters used during training"""

    count_epoch: int = 5
    batch_size: int = 1
    lr: float = 1e-3
    noise_train: Optional[float] = None

    def __str__(self) -> str:
        return f"{self.count_epoch}_{self.batch_size}_{self.lr}_{self.noise_train}"


@dataclass
class ModelParams:
    """Dataset-independent parameters for Main model"""

    conv_out_channels: int = 4
    kernel_size: int = 6
    stride: int = 2
    padding: int = 1
    pool_size: int = 1
    additional_layers: Optional[List[int]] = None

    def get_full_model_params(
        self, in_channels: int, in_size: int, feature_count: int
    ) -> FullModelParams:
        """Convert to FullModelParams"""

        return FullModelParams(
            in_size=in_size,
            in_channels=in_channels,
            conv_out_channels=self.conv_out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            pool_size=self.pool_size,
            feature_count=feature_count,
            additional_layers=self.additional_layers,
        )

    def __str__(self) -> str:
        return (
            f"{self.conv_out_channels}_{self.kernel_size}_{self.stride}_"
            f"{self.padding}_{self.pool_size}_{self.additional_layers}"
        )


def train_and_test(
    dataset_name: str = "MNIST",
    train_params: Optional[TrainParams] = None,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    use_cache: bool = True,
    retrain: bool = False,
    print_rate: Optional[int] = None,
    record: bool = False,
    normalize: bool = False,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.utils.data.DataLoader, str]:
    # pylint:disable=too-many-arguments,too-many-locals,too-many-branches
    """Train and Test the Main model

    This function is based on:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
    """

    if model_params is None:
        model_params = ModelParams()
    if train_params is None:
        train_params = TrainParams()
    if nonidealities is None:
        nonidealities = Nonidealities()
    if normalization is None:
        normalization = Normalization()

    device = get_device()

    (train_dataloader, test_dataloader), dataset_params = get_dataset_and_params(
        name=dataset_name, batch_size=train_params.batch_size
    )

    model = Main(
        model_params.get_full_model_params(*dataset_params),
        nonidealities,
        normalization,
        record or normalize,
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=train_params.lr)

    cache_basename = f"{dataset_name}_{train_params}_{model_params}_{nonidealities}"
    cache_filepath = Path(f"{MODELCACHEDIR}/{cache_basename}.pth")

    if not use_cache or retrain or not cache_filepath.exists():
        for idx_epoch in range(train_params.count_epoch):
            if print_rate is not None:
                logging.info(f"Epoch {idx_epoch+1}/{train_params.count_epoch}:")
            train_model(
                model,
                train_dataloader,
                loss_fn,
                optimizer,
                device=device,
                print_rate=print_rate,
                noise=train_params.noise_train,
            )
            if print_rate is not None and idx_epoch < train_params.count_epoch - 1:
                avg_loss, accuracy = test_model(
                    model, test_dataloader, loss_fn, device=device
                )
                logging.info(f"Average Loss:  {avg_loss:<9f}")
                logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

        if use_cache:
            MODELCACHEDIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.named_state_dict(), cache_filepath)
    else:
        named_state_dict = torch.load(cache_filepath)
        model.load_named_state_dict(named_state_dict)

    if print_rate is not None:
        avg_loss, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
        logging.info(f"Average Loss:  {avg_loss:<9f}")
        logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    if normalize:
        if print_rate is not None:
            logging.info("Normalizing...")
        cache_filepath = Path(f"{MODELCACHEDIR}/{cache_basename}_norm.pth")
        MODELCACHEDIR.mkdir(parents=True, exist_ok=True)
        for layer in model.store.values():
            layer.clear()
        avg_loss, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
        torch.save(
            normalize_values(model.named_state_dict(), model.store), cache_filepath
        )

        if print_rate is not None:
            avg_loss, accuracy = test_model(
                model, test_dataloader, loss_fn, device=device
            )
            logging.info(f"Average Loss:  {avg_loss:<9f}")
            logging.info(f"Accuracy:      {(100*accuracy):<0.4f}%")

    return model, loss_fn, test_dataloader, device


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
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--print_git_info", action="store_true")
    parser.add_argument("--timed", action="store_true")

    return parser.parse_args()


def main() -> None:
    """Main Function"""

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    if args.print_git_info:
        repo = git.Repo(search_parent_directories=True)
        logging.info(f"Git SHA: {repo.head.object.hexsha}")
        diff = repo.git.diff()
        if diff:
            logging.info(repo.git.diff())

    if args.timed:
        start = time.time()

    train_and_test(
        dataset_name=args.dataset_name,
        train_params=TrainParams(
            count_epoch=args.count_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            noise_train=args.noise_train,
        ),
        model_params=ModelParams(
            conv_out_channels=args.conv_out_channels,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            pool_size=args.pool_size,
            additional_layers=args.additional_layers,
        ),
        nonidealities=Nonidealities(
            relu_cutoff=args.relu_cutoff,
            relu_out_noise=args.relu_out_noise,
            linear_out_noise=args.linear_out_noise,
        ),
        normalization=Normalization(
            min_out=args.min_out,
            max_out=args.max_out,
            min_in=args.min_in,
            max_in=args.max_in,
        ),
        use_cache=not args.no_cache,
        retrain=args.retrain,
        print_rate=args.print_rate,
        normalize=args.normalize,
    )

    if args.timed:
        end = time.time()
        logging.info(f"{start} : {end} ({end - start})")


if __name__ == "__main__":
    main()
