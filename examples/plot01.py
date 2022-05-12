"""Modern ROOT Tools example01."""
import itertools
import logging
import os
import pathlib
import shutil
import sys
from mrtools import configuration
from mrtools import model
from mrtools import samplescache
from mrtools import utilities
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple

import click
import ROOT

# For CMSSW ruamel.yaml has been renamed yaml ....
if "CMSSW_BASE" in os.environ:
    import yaml  # type: ignore
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

DEFAULT_OUTDIR = pathlib.Path("/scratch-cbe/users", os.environ["USER"], "example01")

COLOR = [
    ROOT.kRed,
    ROOT.kBlue,
    ROOT.kGreen,
    ROOT.kYellow,
    ROOT.kMagenta,
    ROOT.kTeal,
    ROOT.kAzure,
    ROOT.kCyan,
]

DATA_MARKER = [
    ROOT.kFullCircle,
    ROOT.kFullSquare,
    ROOT.kFullTriangleUp,
    ROOT.kFullTriangleDown,
]

SIGNAL_MARKER = [
    ROOT.kOpenCircle,
    ROOT.kOpenSquare,
    ROOT.kOpenTriangleUp,
    ROOT.kOpenTriangleDown,
]


def define_histos(
    dataframes: Dict[str, Any], histos_defs: List[Dict[str, Any]]
) -> List[Any]:
    """Define histograms for a dataframe."""
    histos = []
    for defs in histos_defs:
        df = dataframes[cast(str, defs["dataframe"])]
        for h1d in defs["Histo1D"]:
            name = h1d["name"]
            title = h1d.get("title", name)
            nbin, xmin, xmax = cast(Tuple[int, float, float], h1d["bins"])
            var = h1d.get("var", name)
            weight = h1d.get("weight")
            if weight:
                histos.append(df.Histo1D((name, title, nbin, xmin, xmax), var, weight))
            else:
                histos.append(df.Histo1D((name, title, nbin, xmin, xmax), var))

    return histos


def make_plots(
    sc: Any, period: str, histos_defs: List[Dict[str, Any]], out: Any
) -> None:

    for defs in histos_def:

        if "data_samples" in defs:
            data_samples = list(sc.find(period, defs["data_samples"]))
        else:
            data_samples = list(sc.list(period, types=model.SampleType.DATA))
    samples_bkg = list(sc.list(period, types=model.SampleType.BACKGROUND))
    samples_sig = list(sc.list(period, types=model.SampleType.SIGNAL))
    for h_defs in histos_defs:
        for h_def in h_defs["Histo1D"]:

            data_samples_names = h_def.get("data_samples")
            if data_samples_names is None:
                data_samples = list(sc.list(period, types=model.SampleType.DATA))
            elif isinstance(data_samples_names, List):
                data_samples = []
                for name in data_samples_names:
                    data_samples += list(sc.find(name))
            else:
                data_samples = list(sc.find(data_samples_name))

            s_bkg = h_def.get("background_samples", samples_bkg)
            s_sig = h_def.get("signal_samples", samples_sig)

            name = h_def["name"]
            title = h_def.get("title", name)

            hs_dat = ROOT.THStack(f"hs_dat_{name}", title)
            for i, s in enumerate(s_dat):
                h = out.Get(f"/{s.name}/{name}")
                if h:
                    h.SetName(f"{s.name}")
                    h.SetMarkerStyle(DATA_MARKER[i])
                    hs_dat.Add(h)
                else:
                    log.warning("Histogram /%s/%s does not exist.", s, name)

            hists = []
            for s in s_bkg:
                h = out.Get(f"/{s.name}/{name}")
                if h:
                    h.SetName(f"{s.name}")
                    hists.append(h)
                else:
                    log.warning("Histogram /%s/%s does not exist.", s, name)

            hs_bkg = ROOT.THStack(f"hs_bkg_{name}", title)
            for i, h in enumerate(sorted(hists, key=lambda x: x.Integral())):
                h.SetFillColor(COLOR[i])
                hs_bkg.Add(h)

            hs_sig = ROOT.THStack(f"hs_sig_{name}", title)
            for i, s in enumerate(s_sig):
                h = out.Get(f"/{s.name}/{name}")
                if h:
                    h.SetName(f"{s.name}")
                    h.SetMarkerStyle(SIGNAL_MARKER[i])
                    h.SetMarkerColor(ROOT.kRed)
                    hs_sig.Add(h)
                else:
                    log.warning("Histogram /%s/%s does not exist.", s, name)

            c1 = ROOT.TCanvas(f"{name}_lin", "c1", 10, 10, 700, 900)
            hs_bkg.Draw("hist")
            hs_dat.Draw("nostack, pe0, same")
            hs_sig.Draw("nostack, p, same")
            c1.Write()
            c2 = ROOT.TCanvas(f"{name}_log", "c2", 10, 10, 700, 900)
            c2.SetLogy()
            hs_bkg.Draw("hist")
            hs_dat.Draw("nostack, pe0, same")
            hs_sig.Draw("nostack, p, same")
            c2.Write()


#            c.SaveAs(f"{name}.png")


def analysis(
    name: str,
    type: model.SampleType,
    attrs: Dict[str, Any],
    urls: Iterator[str],
    tree_name: str,
    histos_def: List[Dict[str, Any]],
) -> List[Any]:
    """Setup analysis."""

    i = 0
    chain = ROOT.TChain(tree_name)
    for url in urls:
        chain.Add(url)
        i += 1

    log.info("Sample %s has %d files", name, i)

    df = ROOT.RDataFrame(chain)

    events = df.Count()

    # Muons and Electrons with medium tights cuts
    df = (
        df.Define("MediumMuon", "Muon_mediumId")
        .Define("MediumMuon_pt", "Muon_pt[MediumMuon]")
        .Define("MediumMuon_phi", "Muon_phi[MediumMuon]")
        .Define("MediumMuon_eta", "Muon_eta[MediumMuon]")
        .Define("MediumMuon_pfRelIso03_all", "Muon_pfRelIso03_all[MediumMuon]")
        .Define("nMediumMuon", "MediumMuon_pt.size()")
        .Define("MediumElectron", "Electron_cutBased >2")
        .Define("MediumElectron_pt", "Electron_pt[MediumElectron]")
        .Define("MediumElectron_phi", "Electron_phi[MediumElectron]")
        .Define("MediumElectron_eta", "Electron_eta[MediumElectron]")
        .Define(
            "MediumElectron_pfRelIso03_all", "Electron_pfRelIso03_all[MediumElectron]"
        )
        .Define("nMediumElectron", "MediumElectron_pt.size()")
    )

    # Muons and electrons with all selection critera
    df = (
        df.Define(
            "GoodMuon",
            "Muon_mediumId && abs(Muon_eta) < 1.5 && Muon_pfRelIso03_all < 0.1",
        )
        .Define("GoodMuon_pt", "Muon_pt[GoodMuon]")
        .Define("GoodMuon_phi", "Muon_phi[GoodMuon]")
        .Define("GoodMuon_eta", "Muon_eta[GoodMuon]")
        .Define("nGoodMuon", "Sum(GoodMuon)")
        .Define("nGoodMuon50", "Sum(Muon_pt[GoodMuon] > 50.)")
        .Define(
            "GoodElectron",
            "Electron_cutBased && abs(Electron_eta) < 1.5 && Electron_pfRelIso03_all < 0.1",
        )
        .Define("GoodElectron_pt", "Electron_pt[GoodElectron]")
        .Define("GoodElectron_phi", "Electron_phi[GoodElectron]")
        .Define("GoodElectron_eta", "Electron_eta[GoodElectron]")
        .Define("nGoodElectron", "Sum(GoodElectron)")
        .Define("nGoodElectron50", "Sum(Electron_pt[GoodElectron] > 50.)")
    )

    # Events with leading muons
    df_muon = (
        df.Filter("nGoodMuon50 > 0 && nGoodElectron50 == 0")
        .Define("MaxGoodMuon_idx", "ArgMax(GoodMuon_pt)")
        .Define("MaxGoodMuon_pt", "GoodMuon_pt[MaxGoodMuon_idx]")
        .Define("MaxGoodMuon_phi", "GoodMuon_phi[MaxGoodMuon_idx]")
        .Define("MaxGoodMuon_eta", "GoodMuon_eta[MaxGoodMuon_idx]")
        .Define("LT", "MaxGoodMuon_pt + met_pt")
    )

    # Events with leading electrons
    df_ele = (
        df.Filter("nGoodElectron50 > 0 && nGoodMuon50 == 0")
        .Define("MaxGoodElectron_idx", "ArgMax(GoodElectron_pt)")
        .Define("MaxGoodElectron_pt", "GoodElectron_pt[MaxGoodElectron_idx]")
        .Define("MaxGoodElectron_phi", "GoodElectron_phi[MaxGoodElectron_idx]")
        .Define("MaxGoodElectron_eta", "GoodElectron_eta[MaxGoodElectron_idx]")
        .Define("LT", "MaxGoodElectron_pt + met_pt")
    )

    histos = define_histos(
        {
            "df": df,
            "df_muon": df_muon,
            "df_ele": df_ele,
        },
        histos_def,
    )

    log.info("Sample %s has %d events", name, events.GetValue())

    return histos


@click.command
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
    "--outdir",
    default=DEFAULT_OUTDIR,
    type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
    help="Output area",
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
    "--small/--no-small", default=False, help="Limit the number of files per sample."
)
def main(
    period: str,
    samples_file: Iterator[pathlib.Path],
    histos_file: pathlib.Path,
    sc_threads: int,
    root_threads: int,
    log_level: str,
    outdir: pathlib.Path,
    small: bool,
):
    """Run example01."""
    utilities.setAllLogLevel(log_level)
    config.load()

    ROOT.gROOT.SetBatch()
    ROOT.EnableImplicitMT(root_threads)

    with samplescache.SamplesCache(sc_threads) as sc:
        for s in samples_file:
            sc.load(s)

        with open(histos_file, "r") as inp:
            histos_defs = yaml.safe_load(inp)

        # outdir.mkdir(parents=True, exist_ok=True)
        # log.info("Writing histos to %s", f"{outdir}/{period}.root")
        # out = ROOT.TFile(f"{outdir}/{period}.root", "RECREATE")
        # for sample in sc.list(period):
        #     urls = (f.url for f in sample)
        #     if small:
        #         urls = itertools.islice(urls, 5)

        #     histos = analysis(
        #         sample.name,
        #         sample.type,
        #         sample.attrs,
        #         urls,
        #         sample.tree_name,
        #         histos_defs,
        #     )
        #     # if ROOT.gDirectory.Get(sample.name) == None:  # noqa: E711
        #     #     ROOT.gDirectory.mkdir(sample.name)
        #     # ROOT.gDirectory.cd(sample.name)
        #     out.mkdir(sample.name)
        #     out.cd(sample.name)
        #     for h in histos:
        #         h.Write()
        #     out.cd()

        shutil.copy(f"{outdir}/{period}.root", f"{outdir}/{period}_plots.root")
        out = ROOT.TFile(f"{outdir}/{period}_plots.root", "UPDATE")

        make_plots(sc, period, histos_defs, out)

        out.Close()
