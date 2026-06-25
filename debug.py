import argparse
import torch


def extract_state_dict(obj):
    """
    Handles common checkpoint formats:
    - raw state_dict
    - checkpoint with 'state_dict'
    - checkpoint with 'model'
    - checkpoint with 'model_state_dict'
    """
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj

    raise ValueError("Unsupported weights file format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="Path to the weights file, e.g. model.pth")
    parser.add_argument(
        "--output",
        default="weight_keys.txt",
        help="Output txt file path",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.weights_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    keys = list(state_dict.keys())

    with open(args.output, "w", encoding="utf-8") as f:
        for key in keys:
            f.write(key + "\n")

    print(f"Saved {len(keys)} keys to {args.output}")


if __name__ == "__main__":
    main()