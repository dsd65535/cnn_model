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

    for layer in model.store:
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{min(vals)},{max(vals)},{mean(vals)},{median(vals)}")

    max0 = max(val for tensor in model.store[0] for val in tensor.flatten().tolist())
    max1 = max(val for tensor in model.store[1] for val in tensor.flatten().tolist())

    state_dict = model.state_dict()
    state_dict["layers.1.bias"] /= max0
    state_dict["layers.1.weight"] /= max0
    state_dict["layers.6.bias"] /= max1
    state_dict["layers.6.weight"] /= max1 / max0
    model.load_state_dict(state_dict)

    for layer in model.store:
        layer.clear()

    _, accuracy = test_model(model, test_dataloader, loss_fn, device=device)
    print(f"New accuracy: {accuracy}")

    for layer in model.store:
        vals = [val for tensor in layer for val in tensor.flatten().tolist()]
        print(f"{min(vals)},{max(vals)},{mean(vals)},{median(vals)}")


if __name__ == "__main__":
    main()
