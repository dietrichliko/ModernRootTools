#!/usr/bin/env python
import json
import os
import pathlib
import sys

import click

PERIODS = [
    "Run2016preVFP",
    "Run2016postVFP",
    "Run2017",
    "Run2018",
]
DEFAULT_OUTPUT = pathlib.Path("/scratch-cbe/users", os.environ["USER"], "MRT")
DEFAULT_NAME = "ex01_{period}.root"

BKG_SAMPLES = ["WJets", "TT", "T", "TTX", "DY", "DYINV", "QCD"]

LUMINOSITY = {
    "Run2016preVFP": 19.5,
    "Run2016postVFP": 16.5,
    "Run2017": 41.48,
    "Run2018": 59.83,
}


@click.command
@click.option(
    "--name",
    default=DEFAULT_NAME,
    help="Name for files",
    show_default=True,
)
@click.option(
    "--period",
    default="Run2017",
    type=click.Choice(PERIODS, case_sensitive=False),
    help="Datataking period",
    show_default=True,
)
@click.option(
    "--output",
    default=DEFAULT_OUTPUT,
    type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
    help="Output directory",
    show_default=True,
)
def main(period: str, name: str, output: pathlib.Path) -> None:
    """Check the event weights."""

    name = name.format(period=period)
    input_path = output.joinpath(name).with_suffix(".json")
    if not input_path.exists():
        sys.exit()
    print(input_path)

    with open(input_path, "r") as input_json:
        stat = json.load(input_json)

    print(stat["SingleMuon"]["yields"])
    for i in range(4):
        name = f"df_muon{i}"
        data = stat["SingleMuon"]["yields"][name]
        mc = sum(stat[k]["yields"][name] for k in BKG_SAMPLES) * LUMINOSITY[period]
        print(f"Dataframe : {name} Data/MC {data/mc:.4f}")
        name = f"df_ele{i}"
        data = stat["SingleElectron"]["yields"][name]
        mc = sum(stat[k]["yields"][name] for k in BKG_SAMPLES) * LUMINOSITY[period]
        print(f"Dataframe : {name} Data/MC {data/mc:.4f}")


if __name__ == "__main__":
    main()
