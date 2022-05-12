"""ModernROOTTools."""
import logging
from mrtools import configuration
from mrtools import samplescache
from mrtools import utilities
from typing import List

import click
import ROOT

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s -  %(name)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mrtools")
config = configuration.get()


@click.command
@click.argument("samples", nargs=-1)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS),
    default="INFO",
    help="Logging levels",
    show_default=True,
)
def main(samples: List[str], log_level: str):
    """Print samples tree."""
    utilities.setAllLogLevel(log_level)
    with samplescache.SamplesCache(1) as sc:

        for sample in samples:
            sc.load(sample)

        sc.print_tree()
