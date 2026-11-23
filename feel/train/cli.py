"""Command-line arguments shared by FEEL training modules."""

import argparse
from pathlib import Path


def existing_directory(value):
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {value}")
    return path.resolve()


def existing_file(value):
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {value}")
    return path.resolve()


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def training_parser(description, output_dir, *, data_dirs, labels=False, epochs=None):
    parser = argparse.ArgumentParser(description=description)
    for name in data_dirs:
        parser.add_argument(
            "--" + name.replace("_", "-"),
            required=True,
            type=existing_directory,
            help=f"Path to the {name.removesuffix('_dir').upper()} directory.",
        )
    if labels:
        parser.add_argument(
            "--labels", required=True, type=existing_file, help="Path to the Excel label workbook."
        )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(output_dir),
        help="Directory for experiment results (default: %(default)s).",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device, such as cpu or cuda:0 (default: auto).",
    )
    if epochs is not None:
        parser.add_argument(
            "--epochs", type=positive_int, default=epochs,
            help="Number of training epochs (default: %(default)s).",
        )
    return parser


def music_emotion_args(description, output_dir, *, data_dirs, epochs, strategies):
    label_dimensions = ("valence", "arousal", "dominance")
    parser = training_parser(
        description, output_dir, data_dirs=data_dirs, labels=True, epochs=epochs
    )
    parser.add_argument(
        "--strategy", action="append", choices=strategies,
        help="Training split to run. Repeat to select multiple splits. The default runs all listed splits.",
    )
    parser.add_argument(
        "--label-dimension", action="append", choices=label_dimensions,
        help="Emotion dimension to run. Repeat to select multiple dimensions. The default runs all three.",
    )
    args = parser.parse_args()
    args.strategy = args.strategy or strategies
    args.label_dimension = args.label_dimension or label_dimensions
    return args


def resolve_device(value):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
