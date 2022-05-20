#!/usr/bin/env python
"""Modern ROOT Tools example01."""
import json
import logging
import os
import pathlib
import time
from mrtools import configuration
from mrtools import model
from mrtools import samplescache
from mrtools import utilities
from typing import Any
from typing import cast
from typing import Iterator
from typing import Tuple

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

ALL_WEIGHTS = [
    "weight",
    "reweightPU",
    "reweightBTag_SF",
    "reweightL1Prefire",
    "reweightLeptonSF",
    "{}",
]


def define_histos(
    dataframes: dict[str, Any], histos_defs: list[dict[str, Any]]
) -> Tuple[list[Any], Any]:
    """Define histograms for a dataframe."""
    histos = []
    event_yields = {}
    for defs in histos_defs:
        df_name = cast(str, defs["dataframe"])
        df = dataframes[df_name]
        weight = defs.get("weight")
        if weight is None:
            event_yields[df_name] = df.Count()
        else:
            event_yields[df_name] = df.Sum(weight)
        for h1d in defs["Histo1D"]:
            name = h1d["name"]
            title = h1d.get("title", name)
            nbin, xmin, xmax = cast(Tuple[int, float, float], h1d["bins"])
            var = h1d.get("var", name)
            if var.isidentifier():
                df1 = df
            else:
                df1 = df.Define(name, var)
                var = name
            if weight:
                h = df1.Histo1D((name, title, nbin, xmin, xmax), var, weight)
            else:
                h = df.Histo1D((name, title, nbin, xmin, xmax), var)
            histos.append(h)

    return histos, event_yields


def analysis(
    period: str,
    sample: model.SampleBase,
    histos_def: list[dict[str, Any]],
    reweight: bool,
    trigger: bool,
    muon_lumi: float,
    ele_lumi: float,
    small: bool,
    muon_trigger: list[str],
    ele_trigger: list[str],
    tight: bool,
) -> Tuple[list[Any], dict[str, Any]]:
    """Setup analysis."""
    nr = len(sample)
    if small:
        chain = sample.chain(5)
        nr = min(nr, 5)
    else:
        chain = sample.chain()
    log.info("Sample %s has %d files", sample.name, nr)

    df = ROOT.RDataFrame(chain)
    events = df.Count()

    # Good Leptons
    if tight:
        df = df.Define(
            "GoodMuon",
            "Muon_tightId && abs(Muon_eta) < 1.5 && Muon_pfRelIso03_all < 0.1",
        ).Define(
            "GoodElectron",
            "Electron_cutBased > 3 && abs(Electron_eta) < 1.5 && Electron_pfRelIso03_all < 0.1",  # noqa: B950
        )
    else:
        df = df.Define(
            "GoodMuon",
            "Muon_mediumId && abs(Muon_eta) < 1.5 && Muon_pfRelIso03_all < 0.1",
        ).Define(
            "GoodElectron",
            "Electron_cutBased > 2 && abs(Electron_eta) < 1.5 && Electron_pfRelIso03_all < 0.1",  # noqa: B950
        )

    # Various leptons
    df = (
        df.Define("MediumMuon_pt", "Muon_pt[Muon_mediumId]")
        .Define("MediumMuon_phi", "Muon_phi[Muon_mediumId]")
        .Define("MediumMuon_eta", "Muon_eta[Muon_mediumId]")
        .Define("TightMuon_pt", "Muon_pt[Muon_tightId]")
        .Define("TightMuon_phi", "Muon_phi[Muon_tightId]")
        .Define("TightMuon_eta", "Muon_eta[Muon_tightId]")
        .Define("GoodMuon_pt", "Muon_pt[GoodMuon]")
        .Define("GoodMuon_phi", "Muon_phi[GoodMuon]")
        .Define("GoodMuon_eta", "Muon_eta[GoodMuon]")
        .Define("MediumElectron_pt", "Electron_pt[Electron_cutBased > 2]")
        .Define("MediumElectron_phi", "Electron_phi[Electron_cutBased > 2]")
        .Define("MediumElectron_eta", "Electron_eta[Electron_cutBased > 2]")
        .Define("TightElectron_pt", "Electron_pt[Electron_cutBased > 3]")
        .Define("TightElectron_phi", "Electron_phi[Electron_cutBased > 3]")
        .Define("TightElectron_eta", "Electron_eta[Electron_cutBased > 3]")
        .Define("GoodElectron_pt", "Electron_pt[GoodElectron]")
        .Define("GoodElectron_phi", "Electron_phi[GoodElectron]")
        .Define("GoodElectron_eta", "Electron_eta[GoodElectron]")
    )

    # Lepton genFlav
    if sample.type != model.SampleType.DATA:
        df = (
            df.Define("MediumMuon_genPartFlav", "Muon_genPartFlav[Muon_mediumId]")
            .Define(
                "MediumElectron_genPartFlav",
                "Electron_genPartFlav[Electron_cutBased > 2]",
            )
            .Define("TightMuon_genPartFlav", "Muon_genPartFlav[Muon_tightId]")
            .Define(
                "TightElectron_genPartFlav",
                "Electron_genPartFlav[Electron_cutBased > 3]",
            )
            .Define("GoodMuon_genPartFlav", "Muon_genPartFlav[GoodMuon]")
            .Define("GoodElectron_genPartFlav", "Electron_genPartFlav[GoodElectron]")
        )

    # Event weights
    if sample.type != model.SampleType.DATA:
        if reweight:
            weights = "*".join(ALL_WEIGHTS)
        else:
            weights = "weight*{}"
        muon_weight = weights.format(muon_lumi)
        ele_weight = weights.format(ele_lumi)
    else:
        muon_weight = "1"
        ele_weight = "1"

    log.debug("Muon weight    : %s", muon_weight)
    log.debug("Electron weight: %s", ele_weight)
    df_muon0 = df.Define("the_weight", muon_weight)
    df_ele0 = df.Define("the_weight", ele_weight)

    if trigger:
        muon_trg = " || ".join(muon_trigger)
        ele_trg = " || ".join(ele_trigger)

        log.debug("Muon trigger    : %s", muon_trg)
        log.debug("Electron trigger: %s", ele_trg)

        df_muon0 = df_muon0.Filter(muon_trigger)
        df_ele0 = df_ele0.Filter(ele_trigger)

    # Events with leading muons
    df_muon1 = (
        df_muon0.Filter("Sum(MediumMuon_pt > 30.) > 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(MediumMuon_pt)")
        .Define("ll_pt", "MediumMuon_pt[ll_idx]")
        .Define("ll_phi", "MediumMuon_phi[ll_idx]")
        .Define("ll_eta", "MediumMuon_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    df_muon2 = (
        df_muon0.Filter("Sum(TightMuon_pt > 30.) > 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(TightMuon_pt)")
        .Define("ll_pt", "TightMuon_pt[ll_idx]")
        .Define("ll_phi", "TightMuon_phi[ll_idx]")
        .Define("ll_eta", "TightMuon_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    df_muon3 = (
        df_muon0.Filter("Sum(GoodMuon_pt > 50.) > 0")
        .Filter("Sum(GoodElectron_pt > 50.) == 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(GoodMuon_pt)")
        .Define("ll_pt", "GoodMuon_pt[ll_idx]")
        .Define("ll_phi", "GoodMuon_phi[ll_idx]")
        .Define("ll_eta", "GoodMuon_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    # Events with leading electrons
    df_ele1 = (
        df_ele0.Filter("Sum(MediumElectron_pt > 30.) > 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(MediumElectron_pt)")
        .Define("ll_pt", "MediumElectron_pt[ll_idx]")
        .Define("ll_phi", "MediumElectron_phi[ll_idx]")
        .Define("ll_eta", "MediumElectron_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    # Events with leading electrons
    df_ele2 = (
        df_ele0.Filter("Sum(TightElectron_pt > 30.) > 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(TightElectron_pt)")
        .Define("ll_pt", "TightElectron_pt[ll_idx]")
        .Define("ll_phi", "TightElectron_phi[ll_idx]")
        .Define("ll_eta", "TightElectron_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    # Events with leading electrons
    df_ele3 = (
        df_ele0.Filter("Sum(GoodElectron_pt > 50.) > 0")
        .Filter("Sum(GoodMuon_pt > 50.) == 0")
        .Filter("HT>200. && met_pt>100.")
        .Define("ll_idx", "ArgMax(GoodElectron_pt)")
        .Define("ll_pt", "GoodElectron_pt[ll_idx]")
        .Define("ll_phi", "GoodElectron_phi[ll_idx]")
        .Define("ll_eta", "GoodElectron_eta[ll_idx]")
        .Define("LT", "ll_pt + met_pt")
        .Define("W_pt", "pt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
        .Define("W_mt", "mt_lep_met(ll_pt, ll_phi, met_pt, met_phi)")
    )

    histos, event_yields = define_histos(
        {
            "df_muon0": df_muon0,
            "df_muon1": df_muon1,
            "df_muon2": df_muon2,
            "df_muon3": df_muon3,
            "df_ele0": df_ele0,
            "df_ele1": df_ele1,
            "df_ele2": df_ele2,
            "df_ele3": df_ele3,
        },
        histos_def,
    )

    start_time = time.time()

    stat = {
        "events": events.GetValue(),
        "yields": {n: y.GetValue() for n, y in event_yields.items()},
        "time": time.time() - start_time,
    }

    return histos, stat


@click.command
@click.option(
    "--name",
    default=DEFAULT_NAME,
    help="Name for output files",
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
    "--sc-threads",
    type=click.IntRange(0, 16),
    default=4,
    help="Number of threads used for staging",
    show_default=True,
)
@click.option(
    "--root-threads",
    type=click.IntRange(0, None),
    default=0,
    help="Number of root threads",
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
@click.option(
    "--small/--no-small",
    default=False,
    help="Limit the number of files per sample.",
    show_default=True,
)
@click.option(
    "--reweight/--no-reweight",
    default=False,
    help="Perform reweighting for various effects.",
    show_default=True,
)
@click.option(
    "--tight/--no-tight",
    default=False,
    help="Tight lepton identification.",
    show_default=True,
)
@click.option(
    "--trigger/--no-trigger",
    default=False,
    help="Poerform trigger selection.",
    show_default=True,
)
def main(
    name: str,
    period: str,
    samples_file: Iterator[pathlib.Path],
    histos_file: pathlib.Path,
    sc_threads: int,
    root_threads: int,
    log_level: str,
    output: pathlib.Path,
    small: bool,
    reweight: bool,
    tight: bool,
    trigger: bool,
):
    """Run example01."""
    utilities.setAllLogLevel(log_level)
    log.setLevel(logging.DEBUG)
    config.load()

    ROOT.gROOT.SetBatch()
    ROOT.EnableImplicitMT(root_threads)

    base_dir = pathlib.Path(__file__).absolute().parent
    ROOT.gROOT.ProcessLine(f'#include "{base_dir}/example01_inc.h"')

    name = name.format(period=period)
    output.mkdir(parents=True, exist_ok=True)

    with samplescache.SamplesCache(sc_threads) as sc:
        for s in samples_file:
            sc.load(s)

        with open(histos_file, "r") as inp:
            histos_defs = yaml.safe_load(inp)

        output_path = output.joinpath(name).with_suffix(".root")
        log.info("Writing histos to %s", output_path)
        output_root = ROOT.TFile(str(output_path), "RECREATE")

        for sample in sc.list(period, types=model.SampleType.DATA):
            if sample.name == "SingleMuon":
                muon_trigger = sample.attrs["trigger"]
                muon_lumi = sample.attrs["integrated_luminosity"]
            elif sample.name == "SingleElectron":
                ele_trigger = sample.attrs["trigger"]
                ele_lumi = sample.attrs["integrated_luminosity"]

        all_stats: dict[str, Any] = {}
        for sample in sc.list(period):
            histos, stat = analysis(
                period,
                sample,
                histos_defs,
                muon_trigger=muon_trigger,
                muon_lumi=muon_lumi,
                ele_trigger=ele_trigger,
                ele_lumi=ele_lumi,
                reweight=reweight,
                tight=tight,
                trigger=trigger,
                small=small,
            )

            log.info(
                "Sample %s has %d events (%.2f kHz)",
                sample.name,
                stat["events"],
                stat["events"] / (stat["time"] * 1e3),
            )

            all_stats[sample.name] = stat
            output_root.mkdir(sample.name)
            output_root.cd(sample.name)
            for h in histos:
                h.Write()
            output_root.cd()

        output_root.Close()

        output_path = output.joinpath(name).with_suffix(".json")
        log.info("Writing stats to %s", output_path)
        with open(output_path, "w") as output_json:
            json.dump(all_stats, output_json)


if __name__ == "__main__":
    main()
