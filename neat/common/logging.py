"""Sets up logging"""

__all__ = ['setup_logging']

import time
import os
import sys
import logging

from pathlib import Path
from typing import Any

LOG_DETAIL = {
    "LOW": "%(levelname)s:%(name)s:%(message)s",
    "MEDIUM": "%(asctime)s:%(levelname)s:%(name)s:%(message)s",
    "HIGH": "%(asctime)s:%(levelname)s:%(name)s:line %(lineno)s:%(message)s",
}


def setup_logging(
        omit_log: bool,
        file: Path,
        severity: str,
        verbosity: str,
        silent_mode: bool = False):

    """
    Configure logging for the run

    :param omit_log: Whether to write a log file
    :param file: Path to the log. Default is <timestamp>_NEAT.log in the current working directory
    :param severity: Severity of events that will be tracked. Defaults to "INFO."
    :param verbosity: Changes the amount of information in the log output.
    :param silent_mode: Default is to output part of the logs to stdout in addition to writing the file
                        Setting this flag will cause it to not print to stdout.
    """

    kwargs: dict[str, Any] = {"force": True, "format": LOG_DETAIL.get(verbosity.upper(), LOG_DETAIL['MEDIUM']),
                              'handlers': []}

    if not silent_mode:
        kwargs['handlers'].append(logging.StreamHandler(sys.stdout))

    if not omit_log:
        kwargs['handlers'].append(logging.FileHandler(file))

    level = getattr(logging, severity.upper(), logging.INFO)
    kwargs['level'] = level

    logging.basicConfig(**kwargs)

    logging.getLogger(__name__).info(f"writing log to: {file}")
