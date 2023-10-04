"""This script test the effects of quantization on accuracy"""
import torch

from cnn_model.__main__ import get_mnist
from cnn_model.basic import get_device
from cnn_model.basic import test_model
from cnn_model.basic import train_model
from cnn_model.models import Ideal


def quantize(val: float, ref: float, bits: int) -> float:
    """Quantize a value to a number of bits not counting sign"""

    ref /= 2**bits

    return round(val / ref) * ref


def main(
    max_bits: int = 7,
    ref: float = 0.5,
    lr: float = 1e-3,
    count_epoch: int = 5,
    batch_size: int = 1,
    bias: bool = False,
) -> None:
    # pylint:disable=too-many-locals,duplicate-code,too-many-arguments
    """Test the effects of quantization"""

    layer_0_params = "layers.0.bias" if bias else "layers.0.weight"
    layer_4_params = "layers.4.bias" if bias else "layers.4.weight"

    device = get_device()

    train_dataloader, test_dataloader = get_mnist(batch_size=batch_size)

    model = Ideal().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for idx_epoch in range(count_epoch):
        print(f"epoch {idx_epoch+1}/{count_epoch}...")
        train_model(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device=device,
        )
    print("model trained...")

    results = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"floats: {results[1]}")

    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    for bits_0 in range(max_bits):
        for bits_4 in range(max_bits):
            # pylint:disable=cell-var-from-loop
            new_state_dict = {k: v.clone() for k, v in original_state_dict.items()}
            new_state_dict[layer_0_params].apply_(
                lambda val: quantize(val, ref, bits_0)
            )
            new_state_dict[layer_4_params].apply_(
                lambda val: quantize(val, ref, bits_4)
            )
            model.load_state_dict(new_state_dict)
            results = test_model(model, test_dataloader, loss_fn, device=device)
            print(f"{bits_0}+{bits_4}: {results[1]}")


if __name__ == "__main__":
    main()
