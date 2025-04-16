# Copyright (c) 2025 lightning-hydra-boilerplate
# Licensed under the MIT License.

"""Logger utilities."""

import logging
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only

# Dictionary to map log levels to the corresponding logger method
LOG_LEVELS = {
    "debug": logging.debug,
    "info": logging.info,
    "warning": logging.warning,
    "error": logging.error,
    "critical": logging.critical,
}


@rank_zero_only
def log_message(level: str, message: str) -> None:
    """Logging function to handle different log levels, ensuring only main process logs.

    Args:
        level (str): Log level (e.g., 'debug', 'info', 'warning', 'error', 'critical').
        message (str): Message to log.
    """
    # Get the appropriate logging function from the dictionary, default to 'info'
    log_func = LOG_LEVELS.get(level, logging.info)
    log_func(message)


def setup_logger(log_file: str | None = None, log_level: int = logging.INFO) -> None:
    """Set up basic logging configuration. Logs will be printed to both console and optionally to a file.

    Args:
        log_file (str | None): Path to the log file. If None, logs only to console.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
