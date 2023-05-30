"""
Plotting for Modern ROOT Tools.
"""
from typing import Any
from typing import Iterator
from typing import cast
import pathlib
import logging
import itertools

import ROOT

from mrtools import datasets

log = logging.getLogger(__name__)


def _split_samples(
    samples_iter: Iterator[datasets.Dataset],
) -> tuple[list[datasets.Dataset], list[datasets.Dataset], list[datasets.Dataset]]:
    dat_samples: list[datasets.Dataset] = []
    bck_samples: list[datasets.Dataset] = []
    sig_samples: list[datasets.Dataset] = []
    for s in samples_iter:
        if s.type == datasets.DatasetType.DATA:
            dat_samples.append(s)
        elif s.type == datasets.DatasetType.BACKGROUND:
            bck_samples.append(s)
        elif s.type == datasets.DatasetType.SIGNAL:
            sig_samples.append(s)
    return dat_samples, bck_samples, sig_samples


def _get_histo(tree: Any, sample: datasets.Dataset, name: str) -> Any:
    h = tree.Get(f'{"_".join(sample.parts[2:])}/{name}')
    h.SetName(sample.name)
    return h


def stack_plot(
    output: pathlib.Path,
    histos_dat: list[Any],
    histos_bkg: list[Any],
    histos_sig: list[Any],
    colors_dat: list[int],
    colors_bkg: list[int],
    colors_sig: list[int],
    title_dat: list[str],
    title_bkg: list[str],
    title_sig: list[str],
    sort_bkg: bool = True,
    rescale: float | bool | None = None,
    x_label: str = "",
    y_label: str = "",
    logx: bool = False,
    logy: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
    marker_size: float = 0.8,
    canvas_size: int | tuple[int, int] = 800,
    ratio_plot: bool = True,
    ratio_height: int = 200,
    ratio_range: tuple[float, float] = (0.1, 1.9),
    ratio_label: str = "",
    legend: bool = True,
    legend_size: tuple[float, float, float, float] = (0.5, 0.6, 0.9, 0.85),
    legend_columns: int = 2,
    format: list[str] | None = None,
):
    if logy:
        name = f"{output.stem}_log"
    else:
        name = f"{output.stem}_lin"
    log.debug("Plotting %s", name)

    # work only on copies of histograms
    h_dat = [h.Clone() for h in histos_dat]
    h_bkg = [h.Clone() for h in histos_bkg]
    h_sig = [h.Clone() for h in histos_sig]
    # default values for colors
    c_dat = [(ROOT.kBlack if c < 0 else c) for c in colors_dat]
    c_bkg = [(i + 2 if c < 0 else c) for i, c in enumerate(colors_bkg)]
    c_sig = [(ROOT.kRed if c < 0 else c) for c in colors_sig]
    t_dat = title_dat.copy()
    t_bkg = title_bkg.copy()
    t_sig = title_sig.copy()

    if sort_bkg:
        h_bkg, c_bkg, t_bkg = cast(
            tuple[list[Any], list[int], list[str]],
            list(
                zip(
                    *sorted(
                        zip(h_bkg, c_bkg, t_bkg, strict=True),
                        key=lambda x: x[0].Integral(),
                    ),
                    strict=True,
                )
            ),
        )

    sum_h_bkg = h_bkg[0].Clone()
    for h in h_bkg[1:]:
        sum_h_bkg.Add(h)

    # setup drawing board
    if not isinstance(canvas_size, tuple):
        canvas_size = (canvas_size, canvas_size)
    c1 = ROOT.TCanvas(name, "", 0, 0, *canvas_size)

    if ratio_plot:
        y_border = ratio_height / c1.GetWindowHeight()
        c1.Divide(1, 2, 0, 0)

        p1 = c1.cd(1)
        p1.SetMargin(0.15, 0.05, 0.1, 0.1)
        p1.SetPad(p1.GetX1(), y_border, p1.GetX2(), p1.GetY2())

        p2 = c1.cd(2)
        p2.SetMargin(0.15, 0.05, 0.4, 0.05)
        p2.SetPad(p2.GetX1(), p2.GetY1(), p1.GetX2(), y_border)
    else:
        p1 = c1
        p1.SetBottomMargin(0.13)
        p1.SetLeftMargin(0.25)
        p1.SetTopMargin(0.07)
        p1.SetRightMargin(0.05)

    if rescale is True:
        sum_dat = max(
            cast(float, h.Integral()) for h in h_dat
        )  # largest data histogram
        sum_bkg = cast(float, sum_h_bkg.Integral())
        try:
            rescale = sum_dat / sum_bkg
            log.debug(
                "Data: %.2f, Background: %.2f, Scale: %.2f", sum_dat, sum_bkg, rescale
            )
        except ZeroDivisionError:
            log.warning("Background histograms empty.")
            rescale = False

    if isinstance(rescale, float):
        log.debug("Rescaling simulation by %f", rescale)
        for h in itertools.chain(h_bkg, h_sig):
            h.Scale(rescale)
        sum_h_bkg.Scale(rescale)

    # min, max
    min_h = min(h.GetMinimum() for h in h_dat + [sum_h_bkg])
    max_h = max(h.GetMaximum() for h in h_dat + [sum_h_bkg])

    # Leave 20% space on top of drawing
    if y_max is None:
        y_max = 10**0.5 * max_h if logy else 1.2 * max_h
    if y_min is None:
        if logy:
            y_min = 0.7
        else:
            y_min = 0.0 if min_h > 0 else 1.2 * min_h

    hstack_dat = ROOT.THStack("hstack_dat", "")
    for i, (h, c) in enumerate(zip(h_dat, c_dat, strict=True)):
        h.SetMarkerSize(marker_size),
        h.SetMarkerStyle(20 + i)
        h.SetMarkerColor(c)
        h.SetTitle("")
        hstack_dat.Add(h)
    hstack_dat.SetMinimum(y_min)
    hstack_dat.SetMaximum(y_max)

    hstack_bkg = ROOT.THStack("hstack_bkg", "")
    for h, c in zip(h_bkg, c_bkg, strict=True):
        h.SetFillColor(c)
        h.SetTitle("")
        hstack_bkg.Add(h)
    hstack_bkg.SetMinimum(y_min)
    hstack_bkg.SetMaximum(y_max)

    hstack_sig = ROOT.THStack("hstack_sig", "")
    for i, (h, c) in enumerate(zip(h_sig, c_sig, strict=True)):
        h.SetMarkerSize(marker_size),
        h.SetMarkerStyle(20 + i)
        h.SetMarkerColor(c)
        h.SetTitle("")
        hstack_sig.Add(h)
    hstack_sig.SetMinimum(y_min)
    hstack_sig.SetMaximum(y_max)
    hstack_sig.SetTitle("")

    p1.cd()
    p1.SetLogx(logx)
    p1.SetLogy(logy)

    hstack_bkg.Draw("hist")
    if not ratio_plot:
        hstack_bkg.GetXaxis().SetTitle(x_label)
    hstack_bkg.GetYaxis().SetTitle(y_label)

    # hstack_bkg.Draw("hist")
    hstack_bkg.Draw("hist")
    hstack_dat.Draw("pe0, nostack, same")
    hstack_sig.Draw("pe0, nostack, same")

    if legend:
        lg = ROOT.TLegend(*legend_size)
        lg.SetNColumns(legend_columns)
        lg.SetFillStyle(0)
        lg.SetShadowColor(ROOT.kWhite)
        lg.SetBorderSize(0)

        lg_dat = zip(h_dat, t_dat, len(h_dat) * ["p"], strict=True)
        lg_bkg = zip(h_bkg, t_bkg, len(h_bkg) * ["f"], strict=True)
        lg_sig = zip(h_sig, t_sig, len(h_sig) * ["p"], strict=True)

        def lg_add_entry(lg: Any, x: Any):
            if x is None:
                lg.AddEntry(ROOT.nullptr, "", "")
            else:
                lg.AddEntry(*x)

        if legend_columns == 2:
            for a, b in itertools.zip_longest(itertools.chain(lg_dat, lg_sig), lg_bkg):
                lg_add_entry(lg, a)
                lg_add_entry(lg, b)
        elif legend_columns == 3:
            for a, b, c in itertools.zip_longest(lg_dat, lg_bkg, lg_sig):
                lg_add_entry(lg, a)
                lg_add_entry(lg, b)
                lg_add_entry(lg, c)
        else:
            for x in itertools.chain.from_iterable(lg_dat, lg_bkg, lg_sig):
                lg.AddEntry(*x)
        lg.Draw()

    if ratio_plot:
        p2.cd()
        p2.SetLogx(logx)

        h_ratio = h_dat[0] / sum_h_bkg
        h_ratio.SetStats(0)
        h_ratio.SetMinimum(ratio_range[0])
        h_ratio.SetMaximum(ratio_range[1])

        h_ratio.SetMarkerSize(marker_size)
        h_ratio.SetMarkerStyle(20)

        label_scale = (canvas_size[1] - ratio_height) / ratio_height
        h_ratio_x = h_ratio.GetXaxis()
        h_ratio_x.SetLabelSize(label_scale * h_ratio_x.GetLabelSize())
        h_ratio_y = h_ratio.GetYaxis()
        h_ratio_y.SetLabelSize(label_scale * h_ratio_y.GetLabelSize())
        h_ratio_y.SetNdivisions(303)

        h_ratio_x.SetTitle(x_label)
        h_ratio_x.SetTitleSize(label_scale * h_ratio_x.GetTitleSize())
        h_ratio_y.SetTitle(ratio_label)
        h_ratio_y.SetTitleSize(label_scale * h_ratio_y.GetTitleSize())
        h_ratio_x.SetTitleOffset(1.5)
        h_ratio_y.SetTitleOffset(0.5)
        h_ratio.Draw("p")

        line = ROOT.TLine(
            h_ratio.GetXaxis().GetXmin(), 1.0, h_ratio.GetXaxis().GetXmax(), 1.0
        )
        line.Draw()

    if format:
        save = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = ROOT.kWarning
        for ext in format:
            c1.SaveAs(str(output.with_name(f"{name}.{ext}")))
        ROOT.gErrorIgnoreLevel = save

    c1.Write()


class StackPlotter:
    histograms: list[Any]
    formats: list[str]
    output: pathlib.Path
    inp_root: Any
    out_root: Any

    def __init__(
        self,
        histograms: list[Any],
        formats: list[str] | None = None,
    ):
        self.histograms = histograms
        self.input = input
        self.formats = formats if formats else ["png"]

    def plot(
        self,
        samples_iter: Iterator[datasets.Dataset],
        output: pathlib.Path,
    ) -> None:
        self.output = output

        samples_dat, samples_bkg, samples_sig = _split_samples(samples_iter)

        log.info("Writing plots to %s", self.output.with_suffix(".plots.root"))
        self.inp_root = ROOT.TFile(str(self.output.with_suffix(".root")), "READ")
        self.out_root = ROOT.TFile(
            str(self.output.with_suffix(".plots.root")), "RECREATE"
        )
        self.output.mkdir(exist_ok=True)

        for h1d in itertools.chain.from_iterable(
            h.get("Histo1D", []) for h in self.histograms
        ):
            if not h1d.get("plot", True):
                log.debug("Skipping plot %s", h1d["name"])
                continue
            self._plot_histo1d(
                h1d,
                samples_dat,
                samples_bkg,
                samples_sig,
            )

        self.inp_root.Close()
        self.out_root.Close()

    def _get_histos(
        self, name, the_samples: list[datasets.Dataset]
    ) -> tuple[list[Any], list[int], list[str]]:
        def color(sample: datasets.Dataset) -> int:
            color = sample.attrs.get("color", -1)
            if isinstance(color, int):
                return color
            elif isinstance(color, str):
                return int(eval(f"ROOT.{color}"))
            else:
                raise ValueError()

        return (
            [self.inp_root.Get(f"{'_'.join(s.parts[2:])}/{name}") for s in the_samples],
            [color(s) for s in the_samples],
            [s.title for s in the_samples],
        )

    def _plot_histo1d(
        self,
        h1d: Any,
        samples_dat: list[datasets.Dataset],
        samples_bkg: list[datasets.Dataset],
        samples_sig: list[datasets.Dataset],
    ) -> None:
        name = cast(str, h1d["name"])
        histos_dat, colors_dat, title_dat = self._get_histos(name, samples_dat)
        histos_bkg, colors_bkg, title_bkg = self._get_histos(name, samples_bkg)
        histos_sig, colors_sig, title_sig = self._get_histos(name, samples_sig)

        for logy in [cast(bool, h1d["logy"])] if "logy" in h1d else [False, True]:
            stack_plot(
                self.output / name,
                histos_dat,
                histos_bkg,
                histos_sig,
                colors_dat,
                colors_bkg,
                colors_sig,
                title_dat,
                title_bkg,
                title_sig,
                cast(bool, h1d.get("sort_bkg", True)),
                cast(float | bool | None, h1d.get("rescale", True)),
                cast(str, h1d.get("title", name)),
                cast(str, h1d.get("label", "Entries")),
                cast(bool, h1d.get("logx", False)),
                logy,
                cast(float | None, h1d.get("y_min")),
                cast(float | None, h1d.get("y_max")),
                cast(float, h1d.get("marker_size", 0.8)),
                cast(int | tuple[int, int], h1d.get("canvas_size", 800)),
                cast(bool, h1d.get("ratio_plot", True)),
                cast(int, h1d.get("ratio_height", 200)),
                cast(tuple[float, float], h1d.get("ratio_range", (0.1, 1.9))),
                cast(str, h1d.get("ratio_label", "MC / Data")),
                cast(bool, h1d.get("legend", True)),
                cast(
                    tuple[float, float, float, float],
                    h1d.get("legend_size", (0.5, 0.6, 0.9, 0.85)),
                ),
                cast(int, h1d.get("legend_columns", 2)),
                self.formats,
            )
