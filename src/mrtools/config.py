"""Configuration for Modern ROOT Tools.

Usage:
   from mrtools import config
   cfg = config.get()
"""
import logging
import os
import pathlib
import shutil
from dataclasses import dataclass
from typing import Any

import tomllib

log = logging.getLogger(__name__)


def _get_boolean(
    data: dict[str, Any], section: str, name: str, default: bool = False
) -> bool:
    """Get boolean value from dict.

    Raises:
      ValueError if value is not boolean
    """
    value = data[section].get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"[{section}]{name} = {value} is not boolean")

    return value


def _get_str(data: dict[str, Any], section: str, name: str, default: str = "") -> str:
    """Get string value from dict.

    Raises:
      ValueError if value is not string
    """
    value = data[section].get(name, default)
    if not isinstance(value, str):
        raise ValueError(f"[{section}]{name} = {value} is not boolean")

    return value


def _get_int(data: dict[str, Any], section: str, name: str, default: int) -> int:
    """Get string value from dict.

    Raises:
      ValueError if value is not string
    """
    value = data[section].get(name, default)
    if not isinstance(value, int):
        raise ValueError(f"[{section}]{name} = {value} is not integer.")

    return value


def _get_path(
    data: dict[str, Any], section: str, name: str, default: str = ""
) -> pathlib.Path:
    """Get path from dict."""

    path = os.path.expandvars(_get_str(data, section, name, default))
    return pathlib.Path(path).expanduser()


def _get_binary(
    data: dict[str, Any], section: str, name: str, default_name: str
) -> pathlib.Path:
    """Get path to binary."""
    value = _get_str(data, section, name)
    if value == "":
        val = shutil.which(default_name)
        if val is None:
            raise ValueError(f"Binary {default_name} not found.")
        path = pathlib.Path(val)
    else:
        if not isinstance(value, str):
            raise ValueError(f"[{section}]{name} = {value} is not a string.")
        path = pathlib.Path(value)
        if not path.exists():
            raise ValueError(f"Binary {value} does not exist.")
        if not os.access(path, os.X_OK):
            raise ValueError(f"File {value} cannot be executed.")

    return path


@dataclass
class Configuration:
    """Configuration for MRTools."""

    stage: bool = False
    store_path: pathlib.Path = pathlib.Path()
    cache_path: pathlib.Path = pathlib.Path()
    samples_cache: pathlib.Path = pathlib.Path()
    output: pathlib.Path = pathlib.Path()
    local_url: str = ""
    global_url: str = ""
    voms_proxy_path: pathlib.Path = pathlib.Path()
    xrdcp: pathlib.Path = pathlib.Path()
    xrdcp_retry: int = 0
    max_xrdcp: int = 0
    max_dasgoclient: int = 0
    voms_proxy_check: bool = False
    voms_proxy_init: pathlib.Path = pathlib.Path()
    voms_proxy_info: pathlib.Path = pathlib.Path()
    dasgoclient: pathlib.Path = pathlib.Path()

    def load(self, config_path: os.PathLike | str) -> None:
        """Load configuration from TOML file."""
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        if "site" not in data:
            raise ValueError("Site section  missing in configuration file.")

        self.stage = _get_boolean(data, "site", "stage", False)
        self.store_path = _get_path(data, "site", "store_path", "")
        self.cache_path = _get_path(data, "site", "cache_path", "")
        self.sample_cache = _get_path(data, "site", "samples_cache", "")
        self.output = _get_path(data, "site", "output")
        self.local_url = _get_str(data, "site", "local_url", "")
        self.global_url = _get_str(data, "site", "global_url", "")
        self.xrdcp_retry = _get_int(data, "site", "xrdcp_retry", 3)
        self.max_xrdcp = _get_int(data, "site", "max_xrdcp", 4)
        self.max_dasgoclient = _get_int(data, "site", "max_dasgoclient", 4)

        self.voms_proxy_check = _get_boolean(data, "cache", "voms_proxy_check", False)
        self.voms_proxy_path = _get_path(data, "cache", "voms_proxy_path", "")

        self.xrdcp = _get_binary(data, "binaries", "xrdcp", "xrdcp")
        self.voms_proxy_init = _get_binary(
            data, "binaries", "voms_proxy_init", "voms-proxy-init"
        )
        self.voms_proxy_info = _get_binary(
            data, "binaries", "voms_proxy_info", "voms-proxy-info"
        )
        self.dasgoclient = _get_binary(data, "binaries", "dasgoclient", "dasgoclient")


_cfg = Configuration()


def get() -> Configuration:
    return _cfg
