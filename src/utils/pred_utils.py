# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Prediction utilities."""

import csv
import json
from pathlib import Path


def save_predictions(predictions: list[dict], output_path: Path, save_format: str = "json") -> None:
    """Save predictions to a file in the specified format.

    Args:
        predictions (list[dict]): A list of prediction dictionaries.
        output_path (Path): Path to the output file.
        save_format (str): Format to save (json or csv).
    """
    if save_format not in {"json", "csv"}:
        error_msg = f"Unsupported save_format: {save_format}"
        raise ValueError(error_msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "json":
        with Path.open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)

    elif save_format == "csv":
        with Path.open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
