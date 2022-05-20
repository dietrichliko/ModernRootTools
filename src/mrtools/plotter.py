"""Moder ROOT Tools Plot Routines."""
import itertools
import logging
import math
import pathlib
from collections.abc import Sequence
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import ROOT

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
    y_min: Optional["float"] = None,
    y_max: Optional["float"] = None,
    sort_background: bool = True,
    marker_size: float = 0.5,
    scale: Optional[float] = None,
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
        histos = sorted(histos_bkg, key=lambda x: x[0].Integral(), reverse=True)
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
