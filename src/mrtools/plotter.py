"""Moder ROOT Tools Plot Routines."""
import itertools
import json
import logging
import math
import pathlib
from collections.abc import Sequence
from mrtools import cache
from mrtools import model
from typing import Any
from typing import Tuple
from typing import Union

import ROOT
import ruamel.yaml

log = logging.getLogger(__name__)


def stackplot(  # noqa: C901
    output: pathlib.Path,
    histos_dat: list[Tuple[Any, str, int]],
    histos_bkg: list[Tuple[Any, str, int]],
    histos_sig: list[Tuple[Any, str, int]],
    canvas_size: Union[int, Tuple[int, int]] = (800, 800),
    logx: bool = False,
    logy: bool = False,
    x_label: str = "",
    y_label: str = "",
    y_min: float = None,
    y_max: float = None,
    sort_background: bool = True,
    marker_size: float = 0.5,
    scale: float = None,
    normalise: bool = False,
    ratio_plot: bool = True,
    ratio_plot_height: int = 200,
    ratio_plot_range: Tuple[float, float] = (0.1, 1.9),
) -> None:
    """Write stackplot."""
    log.debug("Plotting %s", output.stem)

    # setup drawing board
    if isinstance(canvas_size, Sequence):
        c1 = ROOT.TCanvas(output.stem, "", 0, 0, *canvas_size)
    else:
        c1 = ROOT.TCanvas(output.stem, "", 0, 0, canvas_size, canvas_size)

    if ratio_plot:
        y_border = ratio_plot_height / c1.GetWindowHeight()
        c1.Divide(1, 2, 0, 0)

        p1 = c1.cd(1)
        p1.SetBottomMargin(0)
        p1.SetLeftMargin(0.25)
        p1.SetTopMargin(0.07)
        p1.SetRightMargin(0.05)
        p1.SetPad(p1.GetX1(), y_border, p1.GetX2(), p1.GetY2())

        p2 = c1.cd(2)
        p2.SetTopMargin(0)
        p2.SetRightMargin(0.05)
        p2.SetLeftMargin(0.25)
        p2.SetBottomMargin(0.25)
        p2.SetPad(p2.GetX1(), p2.GetY1(), p1.GetX2(), y_border)
    else:
        p1 = c1
        p1.SetBottomMargin(0.13)
        p1.SetLeftMargin(0.15)
        p1.SetTopMargin(0.07)
        p1.SetRightMargin(0.05)

    if normalise:
        sum_dat = max(h[0].Integral() for h in histos_dat)  # largest data histogram
        sum_bkg = sum(h[0].Integral() for h in histos_bkg)  # sum up background
        try:
            scale = sum_dat / sum_bkg
            log.debug(
                "Data: %.2f, Background: %.2f, Scale: %.2f", sum_dat, sum_bkg, scale
            )
        except ZeroDivisionError:
            log.warning("Background histograms empty.")

    if scale is not None:
        log.debug("Rescaling background by %.2f", scale)
        histos_bkg1 = []
        for h in histos_bkg:
            histos_bkg1.append((h[0].Clone(), h[1], h[2]))
        histos_bkg = histos_bkg1
        for h in histos_bkg:
            h[0].Scale(scale)

    hsum_bkg = histos_bkg[0][0].Clone()
    for h in histos_bkg[1:]:
        hsum_bkg.Add(h[0])

    # min, max
    h_min = min(h[0].GetMinimum() for h in histos_dat)
    h_min = min(h_min, hsum_bkg.GetMinimum())
    h_max = max(h[0].GetMaximum() for h in histos_dat)
    h_max = max(h_max, hsum_bkg.GetMaximum())

    # Leave 20% space on top of drawing
    if y_max is None:
        y_max = 10**0.5 * h_max if logy else 1.2 * h_max
    if y_min is None:
        if logy:
            y_min = 0.7
        else:
            y_min = 0.0 if h_min > 0 else 1.2 * h_min

    if ratio_plot:
        if logy:
            y_max = 10 ** (1.3 * (math.log10(y_max) - math.log10(y_min))) + y_min
        else:
            y_max = 1.3 * (y_max - y_min) + y_min

    p1.cd()
    p1.SetLogx(logx)
    p1.SetLogy(logy)

    hs_dat = ROOT.THStack("hs_dat", "")
    for i, h in enumerate(histos_dat):
        h[0].SetMarkerSize(marker_size),
        h[0].SetMarkerStyle(20 + i)
        hs_dat.Add(h[0])
    hs_dat.SetMinimum(y_min)
    hs_dat.SetMaximum(y_max)

    hs_bkg = ROOT.THStack("hs_bkg", "")
    if sort_background:
        histos = sorted(histos_bkg, key=lambda x: x[0].Integral())
    else:
        histos = histos_bkg
    for h in histos:
        if h[2]:
            h[0].SetFillColor(h[2])
        hs_bkg.Add(h[0])
    hs_bkg.SetMinimum(y_min)
    hs_bkg.SetMaximum(y_max)

    hs_sig = ROOT.THStack("hs_sig", "")
    for i, h in enumerate(histos_sig):
        h[0].SetMarkerSize(marker_size),
        h[0].SetMarkerStyle(24 + i)
        if h[2]:
            h[0].SetMarkerColor(h[2])
        else:
            h[0].SetMarkerColor(ROOT.kRed)
        hs_sig.Add(h[0])
    hs_sig.SetMinimum(y_min)
    hs_sig.SetMaximum(y_max)

    hs_bkg.Draw("hist")
    hs_dat.Draw("p nostack same")
    hs_sig.Draw("p nostack same")

    y_label = y_label if y_label else "Entries"
    x_label = x_label if x_label else output.stem

    # precision 3 fonts. see https://root.cern.ch/root/htmldoc//TAttText.html#T5
    for h1 in itertools.chain(hs_bkg, hs_dat, hs_sig):
        x_axis = h1.GetXaxis()
        y_axis = h1.GetYaxis()
        x_axis.SetTitle(x_label)
        y_axis.SetTitle(y_label)
        x_axis.SetTitleFont(43)
        y_axis.SetTitleFont(43)
        x_axis.SetLabelFont(43)
        y_axis.SetLabelFont(43)
        x_axis.SetTitleSize(24)
        y_axis.SetTitleSize(24)
        x_axis.SetLabelSize(20)
        y_axis.SetLabelSize(20)
        x_axis.SetTitleOffset(3.5)
        y_axis.SetTitleOffset(2.5)

    # actual drawing
    hs_bkg.Draw("hist")
    hs_dat.Draw("pe0, nostack, same")
    hs_sig.Draw("pe0, nostack, same")

    # legend

    legend = ROOT.TLegend(0.5, 0.7, 0.9, 0.9)
    legend.SetNColumns(2)
    legend.SetFillStyle(0)
    legend.SetShadowColor(ROOT.kWhite)
    legend.SetBorderSize(0)

    for h in histos_dat:
        legend.AddEntry(h[0], h[1], "p")
    for h in histos_bkg:
        legend.AddEntry(h[0], h[1], "f")
    for h in histos_sig:
        legend.AddEntry(h[0], h[1], "p")

    legend.Draw()

    if ratio_plot:
        p2.cd()
        p2.SetLogx(logx)

        if not isinstance(ratio_plot_range, Sequence):
            ratio_plot_range = (10 ** (-math.log10(ratio_plot_range)), ratio_plot_range)

        h_ratio = histos_dat[0][0] / hsum_bkg
        h_ratio.SetStats(False)
        y_axis = h_ratio.GetYaxis()
        x_axis = h_ratio.GetXaxis()
        y_axis.SetRangeUser(*ratio_plot_range)
        x_axis.SetTitle(x_label)
        y_axis.SetTitle("Data / MC")
        x_axis.SetTitleFont(43)
        y_axis.SetTitleFont(43)
        x_axis.SetLabelFont(43)
        y_axis.SetLabelFont(43)
        x_axis.SetTitleSize(24)
        y_axis.SetTitleSize(24)
        x_axis.SetLabelSize(20)
        y_axis.SetLabelSize(20)
        x_axis.SetTitleOffset(3.5)
        y_axis.SetTitleOffset(2.5)
        x_axis.SetTickLength(0.03 * 3)
        y_axis.SetTickLength(0.03 * 2)
        y_axis.SetNdivisions(505)

        h_ratio.Draw("p")

        line = ROOT.TLine(
            h_ratio.GetXaxis().GetXmin(), 1.0, h_ratio.GetXaxis().GetXmax(), 1.0
        )
        line.Draw()

    save = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    c1.SaveAs(str(output))
    ROOT.gErrorIgnoreLevel = save

    c1.Write()


def tdr_style() -> None:
    """CMS TDR style."""
    style = ROOT.TStyle("tdrStyle", "Style for P-TDR")

    # canvas
    style.SetCanvasBorderMode(0)
    style.SetCanvasColor(ROOT.kWhite)
    style.SetCanvasDefH(600)  # Height of canvas
    style.SetCanvasDefW(600)  # Width of canvas
    style.SetCanvasDefX(0)  # Position on screen
    style.SetCanvasDefY(0)

    # pad
    style.SetPadBorderMode(0)
    # style.SetPadBorderSize(1)
    style.SetPadColor(ROOT.kWhite)
    style.SetPadGridX(False)
    style.SetPadGridY(False)
    style.SetGridColor(0)
    style.SetGridStyle(3)
    style.SetGridWidth(1)

    # For the histo:
    # style.SetHistFillColor(1)
    # style.SetHistFillStyle(0)
    style.SetHistLineColor(1)
    style.SetHistLineStyle(0)
    style.SetHistLineWidth(1)
    # style.SetLegoInnerR(Float_t rad = 0.5)
    # style.SetNumberContours(Int_t number = 20)

    style.SetEndErrorSize(2)
    # style.SetErrorMarker(20)
    style.SetErrorX(0.0)

    style.SetMarkerStyle(20)

    # For the fit/function:
    style.SetOptFit(1)
    style.SetFitFormat("5.4g")
    style.SetFuncColor(2)
    style.SetFuncStyle(1)
    style.SetFuncWidth(1)

    # For the date:
    style.SetOptDate(0)
    # style.SetDateX(Float_t x = 0.01)
    # style.SetDateY(Float_t y = 0.01)

    # For the statistics box:
    style.SetOptFile(0)
    style.SetOptStat(0)  # To display the mean and RMS:   SetOptStat("mr")
    style.SetStatColor(ROOT.kWhite)
    style.SetStatFont(42)
    style.SetStatFontSize(0.025)
    style.SetStatTextColor(1)
    style.SetStatFormat("6.4g")
    style.SetStatBorderSize(1)
    style.SetStatH(0.1)
    style.SetStatW(0.15)
    # style.SetStatStyle(1001)
    # style.SetStatX(0)
    # style.SetStatY(0)

    # Margins:
    style.SetPadTopMargin(0.05)
    style.SetPadBottomMargin(0.13)
    style.SetPadLeftMargin(0.16)
    style.SetPadRightMargin(0.02)

    # For the Global title:

    style.SetOptTitle(0)
    style.SetTitleFont(42)
    style.SetTitleColor(1)
    style.SetTitleTextColor(1)
    style.SetTitleFillColor(10)
    style.SetTitleFontSize(0.05)
    # style.SetTitleH(0) # Set the height of the title box
    # style.SetTitleW(0) # Set the width of the title box
    # style.SetTitleX(0) # Set the position of the title box
    # style.SetTitleY(0.985) # Set the position of the title box
    # style.SetTitleStyle(1001)
    # style.SetTitleBorderSize(2)

    # For the axis titles:

    style.SetTitleColor(1, "XYZ")
    style.SetTitleFont(42, "XYZ")
    style.SetTitleSize(0.06, "XYZ")
    # style.SetTitleXSize(0.02) # Another way to set the size?
    # style.SetTitleYSize(0.02)
    style.SetTitleXOffset(0.9)
    style.SetTitleYOffset(1.25)
    # style.SetTitleOffset(1.1, "Y") # Another way to set the Offset

    # For the axis labels:

    style.SetLabelColor(1, "XYZ")
    style.SetLabelFont(42, "XYZ")
    style.SetLabelOffset(0.007, "XYZ")
    style.SetLabelSize(0.05, "XYZ")

    # For the axis:

    style.SetAxisColor(1, "XYZ")
    style.SetStripDecimals(True)
    style.SetTickLength(0.03, "XYZ")
    style.SetNdivisions(510, "XYZ")
    style.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
    style.SetPadTickY(1)

    # Change for log plots:
    style.SetOptLogx(0)
    style.SetOptLogy(0)
    style.SetOptLogz(0)

    # Postscript options:
    style.SetPaperSize(20.0, 20.0)
    # style.SetLineScalePS(Float_t scale = 3)
    # style.SetLineStyleString(Int_t i, const char* text)
    # style.SetHeaderPS(const char* header)
    # style.SetTitlePS(const char* pstitle)

    # style.SetBarOffset(Float_t baroff = 0.5)
    # style.SetBarWidth(Float_t barwidth = 0.5)
    # style.SetPaintTextFormat(const char* format = "g")
    # style.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
    # style.SetTimeOffset(Double_t toffset)
    # style.SetHistMinimumZero(kTRUE)

    style.cd()

    return style


class SamplesPlotter:
    """Make stackplots of defined histograms."""

    histos_defs: list[dict[str, Any]]
    output: pathlib.Path
    name: str

    def __init__(
        self, histos_file: pathlib.Path, output: pathlib.Path, name: str
    ) -> None:
        """Init SamplePlotter."""
        yaml_parser = ruamel.yaml.YAML(typ="safe")
        with open(histos_file, "r") as inp:
            self.histos_defs = yaml_parser.load(inp)
        self.output = output
        self.name = name

    def plot(self, sc: cache.SamplesCache, period: str) -> None:
        """Draw stack plots for a period."""
        output_path = self.output / f"{self.name}_{period}"
        output_path.mkdir(exist_ok=True)

        inp_root_path = output_path.with_suffix(".root")
        if not inp_root_path.exists():
            log.fatal("Input %s does not exists", inp_root_path)
            return
        log.info("Reading histograms from %s", inp_root_path)
        inp_root = ROOT.TFile(str(inp_root_path), "READ")

        out_root_path = output_path.with_suffix(".plots.root")
        log.info("Writing plots to %s", out_root_path)
        out_root = ROOT.TFile(str(out_root_path), "RECREATE")

        with open(output_path.with_suffix(".json"), "r") as input_json:
            stat = json.load(input_json)

        for h_defs in self.histos_defs:

            df_name = h_defs["dataframe"]
            log.debug("Histograms in dataframe %s", df_name)

            dat_samples = _get_samples(
                sc, period, h_defs.get("data_samples"), model.SampleType.DATA
            )
            bkg_samples = _get_samples(
                sc,
                period,
                h_defs.get("background_samples"),
                model.SampleType.BACKGROUND,
            )
            sig_samples = _get_samples(
                sc, period, h_defs.get("signal_samples"), model.SampleType.SIGNAL
            )

            dat_names = [s.name for s in dat_samples]
            bkg_names = [s.name for s in bkg_samples]
            sig_names = [s.name for s in sig_samples]

            log.debug("Data: %s", ", ".join(dat_names))
            log.debug("Background: %s", ", ".join(bkg_names))
            log.debug("Signal: %s", ", ".join(sig_names))

            nr_events = stat[dat_names[0]][f"{df_name}_nr_events"]
            sum_weights = sum(stat[s][f"{df_name}_sum_weights"] for s in bkg_names)
            scale = nr_events / sum_weights
            log.info(
                "Dataframe %s - Data events %d - Background %.1f - Scale %.3f",
                df_name,
                nr_events,
                sum_weights,
                scale,
            )

            for h1d in h_defs.get("Histo1D", []):
                name = h1d["name"]
                title = h1d.get("title", name)
                histos_dat = _get_histos(inp_root, h1d, dat_samples)
                histos_bkg = _get_histos(inp_root, h1d, bkg_samples)
                histos_sig = _get_histos(inp_root, h1d, sig_samples)
                if not histos_dat:
                    log.error("No data found for %s", name)
                    continue
                if not histos_bkg:
                    log.error("No background found for %s", name)
                    continue
                stackplot(
                    output_path / f"{name}_lin.png",
                    histos_dat,
                    histos_bkg,
                    histos_sig,
                    x_label=title,
                    y_min=h1d.get("ymin_lin"),
                    y_max=h1d.get("ymax_lin"),
                    ratio_plot=h1d.get("ratio_plot", True),
                    scale=scale,
                )

                histos_dat = _get_histos(inp_root, h1d, dat_samples)
                histos_bkg = _get_histos(inp_root, h1d, bkg_samples)
                histos_sig = _get_histos(inp_root, h1d, sig_samples)
                stackplot(
                    output_path / f"{name}_log.png",
                    histos_dat,
                    histos_bkg,
                    histos_sig,
                    logy=True,
                    x_label=title,
                    y_min=h1d.get("ymin_log"),
                    y_max=h1d.get("ymax_log"),
                    ratio_plot=h1d.get("ratio_plot", True),
                    scale=scale,
                )

        inp_root.Close()
        out_root.Close()


def _get_samples(
    sc: cache.SamplesCache,
    period: str,
    pattern: dict[str, Any] | list[str] | str | None = None,
    types: model.SampleTypeSpec = None,
) -> list[model.SampleBase]:

    if isinstance(pattern, dict):
        if period in pattern:
            the_pattern = pattern[period]
        else:
            the_pattern = pattern["default"]
        samples = sc.find(period, the_pattern, types)
    elif pattern is not None:
        samples = sc.find(period, pattern, types)
    else:
        samples = sc.list(period, types=types)
    return list(samples)


def _get_histos(
    tree: Any, hd: dict[str, Any], samples: list[model.SampleBase]
) -> list[Tuple[Any, str, int]]:
    """Read histograms from ROOT tree.

    tree (ROOT.TTree): ROOT TTree
    hid (dict[str, Any]): Histogram definition
    samples (list[model.SampleBase]): :ist of samples

    Returns
        list[Tuple[Any, str, int]] - List of tuples with histogram,
                sample name and sample color
    """
    histos: list[Tuple[Any, str, int]] = []
    name = hd["name"]
    for s in samples:
        subdir = "_".join(s.path.parts[3:])
        h = tree.Get(f"{subdir}/{name}")
        if not h:
            log.error("Histogram %s/%s not found.", subdir, name)
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
