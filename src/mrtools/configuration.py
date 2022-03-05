"""Configuration for Modern ROOT Tools

The configuration is a singleton object.

Usage:

    config = configuration.get_config()
"""
__all__ = ['get_config']

import pathlib
from dataclasses import dataclass
import logging
from typing import Optional, Union

log = logging.getLogger(__package__)

PathOrStr = Union[pathlib.Path,str]

# global variable for singleton 
_config: Optional["Config"] = None

@dataclass
class Binaries:
    """Path for binaries."""
    dasgoclient: pathlib.Path
    voms_proxy_info: pathlib.Path
    voms_proxy_init: pathlib.Path

@dataclass
class SamplesCache:
    """SamplesCache configurations."""

@dataclass
class Site:
    """Site specific configurations."""

class Config:
    """All configurations."""
    bin: Binaries
    sc: SamplesCache
    site: Site

def get_config(config: PathOrStr = "", site: str = "") -> Config:
    """Get Samplescache configuration."""
    global _config
    if _config is not None:
        return _config

    _config = Config(
        bin = Binaries(),
        sc = SamplesCache(),
        site = Site(),
    )

    return _config
