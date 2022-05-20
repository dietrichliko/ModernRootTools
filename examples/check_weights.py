#!/usr/bin/env python
import logging
import os
import pathlib
import sys
from mrtools import utilities
from typing import cast
from typing import Tuple

import click
import ROOT

ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s -  %(name)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mrtools")

NANOTUPLES = pathlib.Path("/scratch-cbe/users/dietrich.liko/StopsCompressed/nanoTuples")

SUBDIRS_MET = {
    "Run2016preVFP": "compstops_UL16APVv9_nano_v7/Met",
    "Run2016postVFP": "compstops_UL16v9_nano_v7/Met",
    "Run2017": "compstops_UL17v9_nano_v7/Met",
    "Run2018": "compstops_UL18v9_nano_v7/Met",
}

SUBDIRS_MET_LEP_ENERGY = {
    "Run2016preVFP": "compstops_UL16APVv9_nano_v6/MetLepEnergy",
    "Run2016postVFP": "compstops_UL16v9_nano_v6/MetLepEnergy",
    "Run2017": "compstops_UL18v9_nano_v6/MetLepEnergy",
    "Run2018": "compstops_UL18v9_nano_v6/MetLepEnergy",
}

PERIODS = [
    "Run2016preVFP",
    "Run2016postVFP",
    "Run2017",
    "Run2018",
]
SKIMS = ["Met", "MetLepEnergy"]

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def scan_weights(dir: pathlib.Path) -> Tuple[int, int, float, float, float]:
    """Scan dataset for event weights."""
    chain = ROOT.TChain("Events")
    nr_files = 0
    for name in os.listdir(dir):
        chain.Add(str(dir / name))
        nr_files += 1

    df = ROOT.RDataFrame(chain)

    nr_events = df.Count()
    sum_weights = df.Sum("weight")
    min_weights = df.Min("weight")
    max_weights = df.Max("weight")

    return (
        nr_files,
        cast(int, nr_events.GetValue()),
        cast(float, sum_weights.GetValue()),
        cast(float, min_weights.GetValue()),
        cast(float, max_weights.GetValue()),
    )


@click.command
@click.option(
    "--period",
    default="Run2017",
    type=click.Choice(PERIODS, case_sensitive=False),
    help="Datataking period",
    show_default=True,
)
@click.option(
    "--skim",
    default="MetLepEnergy",
    type=click.Choice(SKIMS, case_sensitive=False),
    help="Skim",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS),
    default="INFO",
    help="Logging levels",
    show_default=True,
)
@click.option(
    "--root-threads",
    type=click.IntRange(0, None),
    default=0,
    help="Number of root threads",
    show_default=True,
)
def main(period: str, skim: str, log_level: str, root_threads: int) -> None:
    """Check the event weights."""
    utilities.setAllLogLevel(log_level)
    ROOT.gROOT.SetBatch()
    ROOT.EnableImplicitMT(root_threads)

    log.info("Check weight on %s for %s", skim, period)

    if skim == "Met":
        input_path = NANOTUPLES / SUBDIRS_MET[period]
    elif skim == "MetLepEnergy":
        input_path = NANOTUPLES / SUBDIRS_MET_LEP_ENERGY[period]
    else:
        log.fatal("Invalid skim %s", skim)
        sys.exit()

    log.debug("Path %s", input_path)
    for name in sorted(os.listdir(input_path)):
        nr_files, nr_events, sum_weights, min_weights, max_weights = scan_weights(
            input_path / name
        )
        print(
            (
                f"{name:<30s} : {nr_files:3d} : {nr_events:9d} : "
                f"{sum_weights:10.3e} : {min_weights:10.3e} : {max_weights:10.3e}"
            )
        )


if __name__ == "__main__":
    main()
