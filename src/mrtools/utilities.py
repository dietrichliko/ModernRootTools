"""Modern ROOT Tools utility functions.
"""
import logging
import zlib
from mrtools import exceptions
from typing import cast
from typing import Union

from XRootD import client as xrd_client
from XRootD.client.flags import OpenFlags as xrd_OpenFlags


def xrd_checksum(url: str) -> int:
    """Calculate adler32 checksum of file reading its content with XRootD.

    Usage:
        checksum = xrd_checksum("root://eos.grid.vbc.at.at//eos/vbc/...")
    """
    checksum: int = 1
    with xrd_client.File() as f:
        status = f.open(url, xrd_OpenFlags.READ)
        if not status[0].ok:
            raise exceptions.MRTError(status[0].message)
        checksum = 1
        for chunk in f.readchunks():
            checksum = zlib.adler32(chunk, checksum)

    return checksum


def setAllLogLevel(level: Union[str, int]) -> None:
    """The log level of all loggers."""

    log_level = cast(int, getattr(logging, level)) if isinstance(level, str) else level
    for logger in (logging.getLogger(name) for name in logging.root.manager.loggerDict):
        logger.setLevel(log_level)
