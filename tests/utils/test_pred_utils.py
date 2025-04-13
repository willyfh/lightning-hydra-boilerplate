# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Tests for pred_utils.py."""

import csv
import json
from pathlib import Path

import pytest

from utils.pred_utils import save_predictions


@pytest.fixture
def predictions() -> list[dict[str, str | int]]:
    """Fixture providing sample predictions."""
    return [
        {"id": 1, "label": "cat"},
        {"id": 2, "label": "dog"},
    ]


def test_save_predictions_json(tmp_path: Path, predictions: list[dict[str, str | int]]) -> None:
    """Test saving predictions as JSON."""
    output_path = tmp_path / "predictions.json"
    save_predictions(predictions, output_path, save_format="json")

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == predictions


def test_save_predictions_csv(tmp_path: Path, predictions: list[dict[str, str | int]]) -> None:
    """Test saving predictions as CSV."""
    output_path = tmp_path / "predictions.csv"
    save_predictions(predictions, output_path, save_format="csv")

    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert rows == [{"id": "1", "label": "cat"}, {"id": "2", "label": "dog"}]


def test_save_predictions_invalid_format(tmp_path: Path, predictions: list[dict[str, str | int]]) -> None:
    """Test that saving with an unsupported format raises ValueError."""
    output_path = tmp_path / "invalid.out"
    with pytest.raises(ValueError, match="Unsupported save_format"):
        save_predictions(predictions, output_path, save_format="txt")
