import logging
from pathlib import Path
import sys


def setup_logging(
    log_file: Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logging with both file and console handlers.

    Args:
        log_file: Optional path to log file. If provided, logs to both file and stdout.
                 If None, logs only to stdout.
        level: Logging level (default: INFO)
        name: Logger name (default: env_setup_agent)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
