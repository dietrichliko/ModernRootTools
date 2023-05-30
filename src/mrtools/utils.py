"""Various utilities for Modern ROOT Tools."""

import fnmatch
import logging
import types
import importlib.util

import click

# import importlib.util
# import sys
# import types


def click_log_level(logger, *names, **kwargs):
    """Decorator to set logging options with click."""
    if not names:
        names = ["-l", "--log-level"]

    kwargs.setdefault("metavar", "LEVEL")
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("help", "Either CRITICAL, ERROR, WARNING, INFO or DEBUG")
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("multiple", True)

    def decorator(f):
        def _set_level(ctx, param, values):
            for value in values:
                v = value.split(":")
                level = getattr(logging, v[0].upper(), None)
                if level is None:
                    raise click.BadParameter(
                        f"Must be CRITICAL, ERROR, WARNING, INFO or DEBUG, not {v}"
                    )
                if len(v) == 1:
                    logger.setLevel(level)
                else:
                    for name in logging.root.manager.loggerDict:
                        for spec in v[1:]:
                            if fnmatch.fnmatch(name, spec):
                                logging.getLogger(name).setLevel(level)

        return click.option(*names, callback=_set_level, **kwargs)(f)

    return decorator


def human_readable_size(size, decimal_places=2):
    """Readable File size.
    https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    """
    for unit in ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]:
        if size < 1024.0 or unit == "PiB":
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def lazy_import(name: str) -> types.ModuleType:
    spec = importlib.util.find_spec(name)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module
