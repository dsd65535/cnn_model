"""This script normalizes a model"""
from statistics import mean
from statistics import median

from cnn_model.__main__ import train_and_test
from cnn_model.basic import test_model


def main() -> None:
    """Main function"""

    model, loss_fn, test_dataloader, device = train_and_test(record=True)

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"Original accuracy: {accuracy}")

    for name, layer in model.store.items():
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{name}: {min(vals)},{max(vals)},{mean(vals)},{median(vals)}")

    max_conv2d = max(
        val for tensor in model.store["conv2d"] for val in tensor.flatten().tolist()
    )
    max_linear = max(
        val for tensor in model.store["linear"] for val in tensor.flatten().tolist()
    )

    named_state_dict = model.state_dict()
    named_state_dict["conv2d_bias"] /= max_conv2d
    named_state_dict["conv2d_weight"] /= max_conv2d
    named_state_dict["linear_bias"] /= max_linear
    named_state_dict["linear_weight"] /= max_linear / max_conv2d
    model.load_named_state_dict(named_state_dict)

    for layer in model.store.values():
        layer.clear()

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"New accuracy: {accuracy}")

    for name, layer in model.store.items():
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{name}: {min(vals)},{max(vals)},{mean(vals)},{median(vals)}")


if __name__ == "__main__":
    main()
