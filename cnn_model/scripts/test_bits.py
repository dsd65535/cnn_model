"""This script test the effects of quantization on accuracy"""
from cnn_model.__main__ import train_and_test
from cnn_model.basic import test_model


def quantize(val: float, ref: float, bits: int) -> float:
    """Quantize a value to a number of bits not counting sign"""

    ref /= 2**bits

    return round(val / ref) * ref


def main(
    max_bits: int = 7,
    ref: float = 0.5,
    lr: float = 1e-3,
    count_epoch: int = 5,
    dataset_name: str = "MNIST",
    batch_size: int = 1,
    conv_out_channels: int = 32,
    kernel_size: int = 5,
    stride: int = 1,
    padding: int = 0,
    pool_size: int = 2,
    bias: bool = False,
    retrain: bool = False,
) -> None:
    # pylint:disable=too-many-locals,duplicate-code,too-many-arguments
    """Test the effects of quantization"""

    layer_0_params = "layers.0.bias" if bias else "layers.0.weight"
    layer_4_params = "layers.4.bias" if bias else "layers.4.weight"

    model, loss_fn, test_dataloader, device = train_and_test(
        lr=lr,
        count_epoch=count_epoch,
        dataset_name=dataset_name,
        batch_size=batch_size,
        conv_out_channels=conv_out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        retrain=retrain,
    )

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
