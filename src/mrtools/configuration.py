"""Modern ROOT Tools configuration.

The configuration is a singleton object.

Usage:

    configuration.load()
    config = configuration.get()


"""
__all__ = ["get"]

import pathlib
import os
from dataclasses import dataclass
import logging
from typing import Union, Any
import shutil
import sys
import socket

import tomli

log = logging.getLogger(__name__)

PathOrStr = Union[pathlib.Path, str]


def _find_site(data: dict[str, Any]) -> str:
    """Find name of the site from the domainname."""
    domain = socket.getfqdn().split(".", 1)[1]
    for site_name, site_data in data.items():
        if domain in site_data.get("domains", []):
            return site_name

    return ""


def _get_binary_path(data: dict[str, Any], name: str) -> str:
    """If no path is provided, look for the binary in the PATH."""
    path = data.get(name.replace("-", "_"))
    if path is None:
        path = shutil.which(name)
    if path is None:
        log.error("Binary %s not found", name)
        path = ""

    return path


def _get_str(data: dict[str, Any], name: str, default: str = "") -> str:
    """Get string from dictionary."""
    value = data.get(name, default)
    return str(value)


def _get_path(data: dict[str, Any], name: str, default: str = "") -> pathlib.Path:
    """Get path from dictionary."""
    value = data.get(name, default)
    return pathlib.Path(value)


def _get_int(data: dict[str, Any], name: str, default: int = None) -> int:
    """Get int fro  dictionary."""
    value = data.get(name, default)
    return int(value)


def _get_boolean(data: dict[str, Any], name: str, default: bool) -> bool:
    """Get boolean from dictionary."""
    return data.get(name, default)


@dataclass
class Binaries:
    """Path for binaries."""

    dasgoclient: str = ""
    voms_proxy_info: str = ""
    voms_proxy_init: str = ""
    xrdcp: str = ""


@dataclass
class SamplesCache:
    """SamplesCache configurations."""

    threads: int = 0
    xrdcp_retry: int = 0


@dataclass
class Site:
    """Site specific configurations."""

    store_path: pathlib.Path = pathlib.Path("")
    cache_path: pathlib.Path = pathlib.Path("")
    local_url: str = ""
    global_url: str = ""
    stage: bool = True
    batch: str = ""


@dataclass
class Configuration:
    """All configurations."""

    bin: Binaries = Binaries()
    sc: SamplesCache = SamplesCache()
    site: Site = Site()

    def load(self, config_path: PathOrStr = "", site: str = "") -> None:
        """Load configuration data.

        Arguments:
            config_path: Path to configuration file.
                         (default: ~/.config/mrtools/mrtools.toml)
            site: Force a site name.
                  (default: determine the site name from the domainname)

        Raises:
            MRTError if configuration is already loaded
        """
        if config_path == "":
            config_path = (
                pathlib.Path(
                    os.environ.get("XDG_CONFIG_HOME", "~/.config")
                ).expanduser()
                / "mrtools/mrtools.toml"
            )

        elif isinstance(config_path, str):
            config_path = pathlib.Path(config_path)

        if not config_path.exists():
            log.info("Creating configuration file %s", config_path)
            config_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(
                pathlib.Path(__file__).with_name("mrtools.toml"), config_path
            )

        with open(config_path, "rb") as input:
            try:
                config_data = tomli.load(input)
            except tomli.TOMLDecodeError as e:
                log.fatal("Config file %s has invalid format %s", config_path, e)
                sys.exit(1)

        # path to binaries
        cdata = config_data.get("binaries", {})
        self.bin = Binaries(
            dasgoclient=_get_binary_path(cdata, "dasgoclient"),
            voms_proxy_init=_get_binary_path(cdata, "voms-proxy-init"),
            voms_proxy_info=_get_binary_path(cdata, "voms-proxy-info"),
            xrdcp=_get_binary_path(cdata, "xrdcp"),
        )

        # sample_cache
        cdata = config_data.get("samples_cache", {})
        self.sc = SamplesCache(
            xrdcp_retry=_get_int(cdata, "xrdcp_retry", 3),
            threads=_get_int(cdata, "xrdcp_retry", 4),
        )

        # find site name
        cdata = config_data.get("site", {})
        if not site:
            site = _find_site(cdata)
            if not site:
                log.fatal("Could not identify site")
                sys.exit()

        cdata = cdata.get(site)
        self.site = Site(
            stage=_get_boolean(cdata, "stage", False),
            local_url=_get_str(cdata, "local_url"),
            global_url=_get_str(cdata, "global_url"),
            cache_path=_get_path(cdata, "cache_path"),
            store_path=_get_path(cdata, "store_path"),
            batch=_get_str(cdata, "batch", ""),
        )


def get() -> Configuration:
    """Get global configuration object (Singleton).

    Returns:
        configuration dataclass
    """
    return _configuration


_configuration = Configuration()
