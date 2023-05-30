"""Modern ROOT Tools."""

from collections.abc import Iterable
from collections.abc import Iterator
import gzip
import importlib.util
import json
import logging
import pathlib
import sys
import types
from typing import Any, Callable
from typing import cast
import itertools

import correctionlib
import ROOT
import ruamel.yaml as yaml

from mrtools import datasets

RDataFrame = Any
TH1 = Any
CallableDefineDF = Callable[
    [RDataFrame, str, datasets.DatasetType, str], dict[str, RDataFrame]
]

log = logging.getLogger(__name__)


def _ROOT_TChain(tree_name: str, urls: list[str], max_files: int = 0) -> Any:
    """
    Helper routine to create ROOT TChain.

    Attributes:
        tree_name: name of ROOT TTree
        urls: List of urls of ROOT files
        max_files: Limit size of chain
    """
    chain = ROOT.TChain(tree_name)
    url_iter = itertools.islice(urls, max_files) if max_files > 0 else iter(urls)
    for url in url_iter:
        chain.Add(url)
    return chain


class Analysis:
    define_dataframes: CallableDefineDF
    histograms: list[dict[str, Any]]
    working: pathlib.Path
    max_files: int

    def __init__(
        self,
        define_dataframes: CallableDefineDF,
        histograms: list[Any],
        working: pathlib.Path,
        max_files: int = 0,
    ) -> None:
        self.define_dataframes = define_dataframes
        self.histograms = histograms
        self.working = working
        self.max_files = max_files

    def map(
        self,
        dataset_chain: tuple[str, list[str]],
        dataset_name: str,
        dataset_type: datasets.DatasetType,
        period: str,
    ) -> tuple[dict[str, TH1], dict[str, float | int]]:
        log.debug("Mapping %s ...", dataset_name)
        chain = _ROOT_TChain(*dataset_chain, self.max_files)
        df_main = ROOT.RDataFrame(chain)
        df: dict[str, Any] = self.define_dataframes(
            df_main, dataset_name, dataset_type, period
        )
        histos, counters = self.book_histograms(df_main, df)
        ROOT.RDF.RunGraphs(list(histos.values()) + list(counters.values()))

        log.debug("Events: %d", counters["events"].GetValue())
        return (
            {name: hist.GetValue() for name, hist in histos.items()},
            {name: value.GetValue() for name, value in counters.items()},
        )

    def book_histograms(
        self, df_main: RDataFrame, dataframes: dict[str, RDataFrame]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        histos: dict[str, Any] = {}
        counters: dict[str, Any] = {}
        for defs in self.histograms:
            df_name = defs["dataframe"]
            if df_name not in dataframes:
                log.error("RDataframe %s not found.", df_name)
                continue
            df_weight = str(defs.get("weight", "1."))

            log.debug(
                'Booking histograms for dataframe "%s" with weight "%s"',
                df_name,
                df_weight,
            )

            for h1d in defs.get("Histo1D", []):
                name = h1d["name"]
                title = h1d.get("title", name)
                var = h1d.get("var", name)
                weight = h1d.get("weight", df_weight)
                if "bins" in h1d:
                    nbins, xmin, xmax = h1d["bins"]
                    log.debug(
                        'Histo1D(("%s","%s",%d, %f,%f),%s,%s)',
                        name,
                        title,
                        nbins,
                        xmin,
                        xmax,
                        var,
                        weight,
                    )
                    histos[name] = dataframes[df_name].Histo1D(
                        (name, title, nbins, xmin, xmax), var, weight
                    )

        counters = {f"{n}_events": df.Count() for n, df in dataframes.items()}
        counters["events"] = df_main.Count()

        return histos, counters

    def reduce(
        self,
        sample: datasets.Dataset,
        histos: list[dict[str, Any]],
        counters: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        log.debug("Reducing %s ...", sample)
        merged_histos: dict[str, Any] = {n: h.Clone() for n, h in histos[0].items()}
        for hs in histos[1:]:
            for n, h in hs.items():
                merged_histos[n] = merged_histos[n] + h

        merged_counters = counters[0].copy()
        for cs in counters[1:]:
            for n, c in cs.items():
                if n.startswith("max_"):
                    merged_counters[n] = max(merged_counters[n], c)
                elif n.startswith("min_"):
                    merged_counters[n] = min(merged_counters[n], c)
                else:
                    merged_counters[n] += c

        log.debug("Events: %d", merged_counters["events"])
        return merged_histos, merged_counters


class Processor:
    analysis: Analysis

    def __init__(
        self,
        define_dataframes: CallableDefineDF,
        histograms: list[Any],
        root_threads: int = 0,
        max_files: int = 0,
    ) -> None:
        ROOT.gROOT.SetBatch()
        if root_threads >= 0:
            ROOT.EnableImplicitMT(root_threads)
        correctionlib.register_pyroot_binding()

        self.analysis = Analysis(
            define_dataframes, histograms, pathlib.Path(), max_files
        )

    def run(
        self,
        the_datasets: Iterable[datasets.Dataset],
        output: pathlib.Path,
    ) -> None:
        all_histos: dict[str, dict[str, Any]] = {}
        all_counters: dict[str, dict[str, Any]] = {}
        for dataset in the_datasets:
            log.info("Dataset %s", dataset)
            if isinstance(dataset, datasets.DatasetFrom):
                hs, cs = self.analysis.map(
                    (dataset.tree_name, [f.url() for f in dataset]),
                    dataset.name,
                    dataset.type,
                    dataset.period,
                )
                all_histos[str(dataset)] = hs
                all_counters[str(dataset)] = cs
            else:
                for parent, groups, children in dataset.walk(topdown=False):
                    histos: list[dict[str, Any]] = []
                    counters: list[dict[str, Any]] = []
                    for ds in children:
                        hs, cs = self.analysis.map(
                            (ds.tree_name, [f.url() for f in dataset]),
                            ds.name,
                            ds.type,
                            ds.period,
                        )
                        histos.append(hs)
                        counters.append(cs)
                        all_histos[str(ds)] = hs
                        all_counters[str(ds)] = cs
                    for ds_name in map(str, groups):
                        histos.append(all_histos[ds_name])
                        counters.append(all_counters[ds_name])
                    hs, cs = self.analysis.reduce(parent, histos, counters)
                    all_histos[str(parent)] = hs
                    all_counters[str(parent)] = cs
        root_file = output.with_suffix(".root")
        log.info("Writing %s ...", root_file)
        out_root = ROOT.TFile.Open(str(root_file), "RECREATE")
        for sample_path, sample_histos in all_histos.items():
            subdir = "_".join(sample_path.split("/")[3:])
            ROOT.gDirectory.mkdir(subdir)
            ROOT.gDirectory.cd(subdir)
            for h in sample_histos.values():
                h.Write()
            ROOT.gDirectory.cd("/")
        out_root.Close()

        json_file = output.with_suffix(".json.gz")
        log.info("Writing %s ...", json_file)
        with gzip.open(json_file, mode="wt", encoding="UTF-8") as out_json:
            json.dump(all_counters, out_json)


def load_module_from_file(analysis: pathlib.Path) -> types.ModuleType:
    """Initialise user analysis."""

    sys.path.insert(0, str(analysis.parent))
    spec = importlib.util.spec_from_file_location("user", analysis)
    if spec is None:
        log.fatal("Could not load %s", analysis)
        sys.exit(1)

    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)  # type: ignore

    return user_module


def get_files_from_module(
    module: types.ModuleType, name: str, base_dir: pathlib.Path
) -> list[pathlib.Path]:
    """Get samples and histogram definitions."""
    try:
        defs = getattr(module, name)
    except AttributeError:
        return []

    defs_path: list[pathlib.Path] = []
    if isinstance(defs, pathlib.Path | str):
        defs_path.append(pathlib.Path(base_dir, defs))
    elif isinstance(defs, Iterable):
        for d in defs:
            pathlib.Path(base_dir, d)
            if isinstance(d, pathlib.Path | str):
                defs_path.append(pathlib.Path(base_dir, d))
            else:
                raise AttributeError(f"Element {d} of attribute {name} is no path.")
    else:
        raise AttributeError(f"Attribute {name} is no path or list of paths.")

    return defs_path


def get_list_from_module(module: types.ModuleType, name: str) -> list[str]:
    try:
        value = getattr(module, name)
    except AttributeError:
        return []

    if isinstance(value, str):
        return [value]
    elif isinstance(value, Iterable):
        for n in value:
            if not isinstance(n, str):
                raise AttributeError("Item %s of %s is not string.", n, name)
        return cast(list[str], value)

    raise AttributeError(f"user.{name} has to be a string or a list of strings.")


def find_samples(sc, period: str, names: list[str]) -> Iterator[datasets.Dataset]:
    for name in names:
        sample_list = list(sc.find(period, name))
        if not sample_list:
            log.error("Sample %s not found.", name)
        elif len(sample_list) > 1:
            log.warning("Sample %s is not unique.", name)
        for sample in sample_list:
            yield sample


def load_histos(file_name: pathlib.Path) -> list[Any]:
    log.info("Loading histos from %s", file_name)
    yaml_parser = yaml.YAML(typ="safe")
    with open(file_name, "r") as inp:
        data = yaml_parser.load(inp)

    return cast(list[Any], data)
