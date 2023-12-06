"""This script test the effects of quantization on accuracy"""
from typing import Optional

from cnn_model.__main__ import ModelParams
from cnn_model.__main__ import train_and_test
from cnn_model.__main__ import TrainParams
from cnn_model.basic import test_model
from cnn_model.models import Nonidealities
from cnn_model.models import Normalization


def quantize(val: float, ref: float, bits: int) -> float:
    """Quantize a value to a number of bits not counting sign"""

    step = ref / 2**bits

    return max(
        min((round(val / step + 0.5) - 0.5) * step, ref - step / 2), -ref + step / 2
    )


def main(
    max_bits: int = 7,
    ref: float = 0.5,
    bias: bool = False,
    dataset_name: str = "MNIST",
    train_params: Optional[TrainParams] = None,
    model_params: Optional[ModelParams] = None,
    nonidealities: Optional[Nonidealities] = None,
    normalization: Optional[Normalization] = None,
    use_cache: bool = True,
    retrain: bool = False,
    print_rate: Optional[int] = None,
) -> None:
    # pylint:disable=too-many-locals,duplicate-code,too-many-arguments
    """Test the effects of quantization"""

    layer_conv_params = "layers.1.bias" if bias else "layers.1.weight"
    layer_fc_params = "layers.5.bias" if bias else "layers.5.weight"

    if model_params is None:
        model_params = ModelParams()

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

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"unlimited floats: {accuracy}")

    original_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    original_state_dict[layer_conv_params] = original_state_dict[
        layer_conv_params
    ].apply_(lambda val: min(ref, max(-ref, val)))
    original_state_dict[layer_fc_params] = original_state_dict[layer_fc_params].apply_(
        lambda val: min(ref, max(-ref, val))
    )

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"lmiited floats: {accuracy}")

    print("," + ",".join(str(bits_fc) for bits_fc in range(max_bits)))
    for bits_conv in range(max_bits):
        results = []
        for bits_fc in range(max_bits):
            # pylint:disable=cell-var-from-loop
            new_state_dict = {k: v.clone() for k, v in original_state_dict.items()}
            new_state_dict[layer_conv_params] = new_state_dict[
                layer_conv_params
            ].apply_(lambda val: quantize(val, ref, bits_conv))
            new_state_dict[layer_fc_params] = new_state_dict[layer_fc_params].apply_(
                lambda val: quantize(val, ref, bits_fc)
            )
            model.load_state_dict(new_state_dict)
            _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
            results.append(str(accuracy))
        print(f"{bits_conv}," + ",".join(results))


if __name__ == "__main__":
    main()
