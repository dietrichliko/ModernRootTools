"""CLI for Modern ROOT Tools."""

import click
import os
import pathlib
import logging
import shutil
import importlib.resources


from mrtools import config
from mrtools import utils

from mrtools.commands.list import list_datasets
from mrtools.commands.run import run
from mrtools.commands.plot import plot
from mrtools.commands.prun import prun
from mrtools.commands.new import new
from mrtools.commands.fix import fix

# ROOT = utils.lazy_import("ROOT")


logging.basicConfig(
    format="%(asctime)s - %(levelname)-8s - %(name)-16s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

log_main = logging.getLogger(__package__)
log = logging.getLogger(__name__)
cfg = config.get()

DEFAULT_CONFIG_PATH = os.path.expanduser(
    f'{os.environ.get("XDG_CONFIG_HOME", "~/.config")}' "/mrtools/mrtools.toml"
)


@click.group(context_settings=dict(max_content_width=120))
@click.option(
    "--config",
    default=DEFAULT_CONFIG_PATH,
    type=click.Path(dir_okay=False, path_type=pathlib.Path),
    show_default=True,
    help="Configuration",
)
@utils.click_log_level(log_main)
def main(config: pathlib.Path) -> None:
    """Modern ROOT Tools.

    A framework for the analysis of CMS datasets based on RDataFrame using
    dask for distributed analysis.
    """
    if not config.exists():
        log.info("Creating configuration file %s", config)
        config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        new_cfg = importlib.resources.files(__package__).joinpath("mrtools.toml")
        with importlib.resources.as_file(new_cfg) as new_cfg_path:
            shutil.copyfile(new_cfg_path, config)

    cfg.load(config)


main.add_command(list_datasets)
main.add_command(run)
main.add_command(plot)
main.add_command(prun)
main.add_command(new)
main.add_command(fix)
