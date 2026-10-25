"""Structured, machine-readable diagnostics without contaminating CLI JSON output."""

import json
import logging
import sys
import time

logger = logging.getLogger("erag")


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)


def event(name: str, **fields) -> None:
    logger.info(json.dumps({"event": name, "timestamp": time.time(), **fields}, sort_keys=True))
