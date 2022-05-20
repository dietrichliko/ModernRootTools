#!/usr/bin/env python
"""Modern ROOT Tools example01."""
import logging
import os
import pathlib
import sys
from mrtools import configuration
from mrtools import model
from mrtools import plotter
from mrtools import samplescache
from mrtools import utilities
from typing import Any
from typing import Iterator
from typing import Tuple
from typing import Union

import click
import ROOT

# For CMSSW ruamel.yaml has been renamed yaml ....
# TODO: Find something better
if "CMSSW_BASE" in os.environ:
    #    import yaml
    pass
else:
    import ruamel.yaml as yaml

ROOT.PyConfig.IgnoreCommandLineOptions = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s -  %(name)s - %(message)s",
    datefmt="%y-%m-%d %H:%M:%S",
)
log = logging.getLogger("mrtools")
config = configuration.get()

PERIODS = [
    "Run2016preVFP",
    "Run2016postVFP",
    "Run2017",
    "Run2018",
]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

DEFAULT_OUTPUT = pathlib.Path("/scratch-cbe/users", os.environ["USER"], "MRT")
DEFAULT_NAME = "example01_{period}.root"


def get_samples(
    sc: Any,
    period: str,
    sample_names: Union[None, str, list[str]],
    types: model.SampleType,
) -> list[model.Sample]:
    """Get samples per name."""
    if sample_names:
        samples = sc.find(period, sample_names)
    else:
        samples = sc.list(period, types=types)

    return list(samples)


def get_histos(
    tree: Any, h_def: dict[str, Any], samples: list[model.Sample]
) -> list[Tuple[Any, str, int]]:
    """Get histograms from file."""
    histos: list[Tuple[Any, str, int]] = []
    name = h_def["name"]
    for s in samples:
        h = tree.Get(f"{s.name}/{name}")
        if not h:
            log.error("Histogram /%s/%s not found.", s.name, name)
            continue
        h.SetName(s.name)
        if "color" in s.attrs:
            color = s.attrs["color"]
            if isinstance(color, str) and color[0] == "k":
                color = eval(f"ROOT.{color}")
            color = int(color)
        else:
            color = None

        histos.append((h, s.title, color))

    return histos


def make_plots(
    sc: Any,
    period: str,
    histos_defs: list[dict[str, Any]],
    name: str,
    output: pathlib.Path,
) -> None:
    """Make plots."""
    log.info("make_plots")

    utilities.tdr_style()
    ROOT.gROOT.ForceStyle()
    name = name.format(period=period)
    input_path = output.joinpath(name).with_suffix(".root")
    if not input_path:
        log.fatal("Input %s does not exists", input_path)
        sys.exit()
    log.info("Reading histograms from %s", input_path)
    input_root = ROOT.TFile(str(input_path), "READ")

    output_dir = output.joinpath(name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for defs in histos_defs:

        df_name = defs["dataframe"]
        log.debug("Histograms in dataframe %s", df_name)

        samples_dat = get_samples(
            sc, period, defs.get("data_samples"), model.SampleType.DATA
        )
        samples_bkg = get_samples(
            sc, period, defs.get("background_samples"), model.SampleType.BACKGROUND
        )
        samples_sig = get_samples(
            sc, period, defs.get("signal_samples"), model.SampleType.SIGNAL
        )

        log.debug("Data: %s", ", ".join(s.name for s in samples_dat))
        log.debug("Background: %s", ", ".join(s.name for s in samples_bkg))
        log.debug("Signal: %s", ", ".join(s.name for s in samples_sig))

        for h_def in defs.get("Histo1D", []):
            name = h_def["name"]
            title = h_def.get("title", name)
            histos_dat = get_histos(input_root, h_def, samples_dat)
            histos_bkg = get_histos(input_root, h_def, samples_bkg)
            histos_sig = get_histos(input_root, h_def, samples_sig)
            plotter.stackplot(
                output_dir / f"{name}_lin.png",
                histos_dat,
                histos_bkg,
                histos_sig,
                x_label=title,
                # ratio_plot=False,
            )
            histos_dat = get_histos(input_root, h_def, samples_dat)
            histos_bkg = get_histos(input_root, h_def, samples_bkg)
            histos_sig = get_histos(input_root, h_def, samples_sig)
            plotter.stackplot(
                output_dir / f"{name}_log.png",
                histos_dat,
                histos_bkg,
                histos_sig,
                logy=True,
                x_label=title,
                # ratio_plot=False,
            )


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
    "--samples-file",
    type=click.Path(exists=True, resolve_path=True),
    default=["examples/samples/MetLepEnergy_nanoNtuple_v6.yaml"],
    multiple=True,
    help="Sample definitions",
    show_default=True,
)
@click.option(
    "--histos-file",
    type=click.Path(exists=True, resolve_path=True),
    default="examples/example01.histos.yaml",
    help="Histogram definitions",
    show_default=True,
)
@click.option(
    "--output",
    default=DEFAULT_OUTPUT,
    type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
    help="Output directory",
    show_default=True,
)
@click.option(
    "--log-level",
    type=click.Choice(LOG_LEVELS),
    default="INFO",
    help="Logging levels",
    show_default=True,
)
def main(
    name: str,
    period: str,
    samples_file: Iterator[pathlib.Path],
    histos_file: pathlib.Path,
    output: pathlib.Path,
    log_level: str,
):
    """Run example01."""
    utilities.setAllLogLevel(log_level)
    config.load()

    ROOT.gROOT.SetBatch()

    with samplescache.SamplesCache() as sc:
        for s in samples_file:
            sc.load(s)

        with open(histos_file, "r") as inp:
            histos_defs = yaml.safe_load(inp)

        make_plots(sc, period, histos_defs, name, output)


if __name__ == "__main__":
    main()
