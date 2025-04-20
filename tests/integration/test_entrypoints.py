# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Integration tests for train, eval, and predict."""

import subprocess  # noqa: S404
import sys
from pathlib import Path


def run_train(tmp_path: Path) -> Path:
    """Run the training script and return the path to the best checkpoint.

    Args:
        tmp_path (Path): Temporary path to store the training run.

    Returns:
        Path: The path to the best checkpoint.
    """
    run_dir = tmp_path / "train_run"
    best_ckpt_path = run_dir / "checkpoints" / "best-checkpoint.ckpt"

    # Command split across multiple lines for readability
    command = [
        sys.executable,
        "src/train.py",
        "experiment=example_experiment",
        f"hydra.run.dir={run_dir}",
    ]

    # Ensure no untrusted input is being used
    assert all(isinstance(arg, str) for arg in command), "Command arguments must be strings"
    train_result = subprocess.run(command, text=True, check=True)  # noqa: S603

    assert train_result.returncode == 0
    return best_ckpt_path


def run_predict(tmp_path: Path, best_ckpt_path: Path) -> None:
    """Run the prediction script.

    Args:
        tmp_path (Path): Temporary path to store the prediction run.
        best_ckpt_path (Path): Path to the best checkpoint file.
    """
    run_dir = tmp_path / "train_run"

    # Command split across multiple lines for readability
    command = [
        sys.executable,
        "src/predict.py",
        "experiment=example_experiment",
        f"ckpt_path={best_ckpt_path}",
        f"hydra.run.dir={run_dir}",
    ]

    predict_result = subprocess.run(command, text=True, check=True)  # noqa: S603

    assert predict_result.returncode == 0


def run_eval(tmp_path: Path, best_ckpt_path: Path) -> None:
    """Run the evaluation script.

    Args:
        tmp_path (Path): Temporary path to store the evaluation run.
        best_ckpt_path (Path): Path to the best checkpoint file.
    """
    run_dir = tmp_path / "train_run"

    # Command split across multiple lines for readability
    command = [
        sys.executable,
        "src/eval.py",
        "experiment=example_experiment",
        f"ckpt_path={best_ckpt_path}",
        f"hydra.run.dir={run_dir}",
    ]

    eval_result = subprocess.run(command, text=True, check=True)  # noqa: S603

    assert eval_result.returncode == 0


def test_entrypoint_execution(tmp_path: Path) -> None:
    """Test the execution of the training, prediction, and evaluation scripts.

    Args:
        tmp_path (Path): Temporary path for storing test files.
    """
    best_ckpt_path = run_train(tmp_path)
    run_predict(tmp_path, best_ckpt_path)
    run_eval(tmp_path, best_ckpt_path)
