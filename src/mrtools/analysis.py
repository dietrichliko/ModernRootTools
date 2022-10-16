"""Modern ROOT Tools."""
import abc
import json
import logging
import pathlib
import sys
import time
from array import array
from mrtools import cache
from mrtools import config
from mrtools import model
from typing import Any
from typing import cast
from typing import Generic
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import click
from ruamel.yaml import YAML

# suppress FutureWarning from dask
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore", FutureWarning, 20)


import dask.distributed as dd
import dask_jobqueue
import ROOT

log = logging.getLogger(__name__)
cfg = config.get()

_click_options: dict[str, Any] = {}

SMALL_MAX_FILES = 3

DataFrame = Any  # ROOT DataFrame
DictHisto = dict[str, Any]  # ROOT Histograms
DictValue = dict[str, int | float]
T = TypeVar("T")

class WorkerPlugin(dd.WorkerPlugin):
    """Initialise the Dask Worker.

    Initialization of Logging, Configuration and ROOT on the
    worker processes.
    """

    cfg: config.Configuration
    log_level: int
    _user_proxy: str
    root_threads: int
    root_includes: Sequence[str]

    def __init__(self, root_threads: int, root_includes: Sequence[str]) -> None:
        """Init WorkerPlugin."""
        self.cfg = cfg
        self.log_level = log.getEffectiveLevel()
        self.root_threads = root_threads
        self.root_includes = root_includes

    def setup(self, worker: dd.Worker) -> None:
        """Worker init."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s -  %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
        )
        for logger in (
            logging.getLogger(name) for name in logging.root.manager.loggerDict
        ):
            logger.setLevel(self.log_level)

        global cfg
        cfg = self.cfg

        ROOT.gROOT.SetBatch()
        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gErrorIgnoreLevel = ROOT.kError
        ROOT.EnableImplicitMT(self.root_threads)
        for root_incl in self.root_includes:
            ROOT.gROOT.ProcessLine(f'#include "{root_incl}"')


class Analysis(Generic[T], abc.ABC):
    """Base class for distributed analysis."""

    @abc.abstractmethod
    def map(self, sample: model.Sample) -> T:
        """Transform sample data to results.

        Args:
            sample (model.Sample): Sample to be transformed.

        Returns:
            T are the results of the mapping.
        """
        pass

    @abc.abstractmethod
    def reduce(self, sample: model.SampleBase, results: list[T]) -> T:
        """Combine the results of various SampleGroups.

        Args:
            sample (model.SampleBase): The sample to combine.
            results (list[T]): Results obtained by the children.

        Returns:
            T are the combined results.
        """
        pass

    def gather(self, future_to_sample: dict[dd.Future, model.SampleBase]) -> None:
        """Wait for all results.

        The default implentation is just waiting for the jobs to finish.

        Args:
            future_to_sample (dict[dd.Future, model.SampleBase]): Mapping
                of futures to samples
        """
        dd.wait(future_to_sample.keys())


class HistoAnalysis(Analysis):
    """Abstract base class for Histogram Analysis."""

    _histo_defs: Any
    _output: pathlib.Path
    _small: bool

    def __init__(
        self,
        histo_file: pathlib.Path,
        output: pathlib.Path,
        small: bool,
    ) -> None:
        """Init HistoAnalysis.

        Args:
            histo_file (pathlib.Path): histogram definitions
            output (pathlib.Path): path for ouput files
            small (bool): reduce samples size
        """
        with open(histo_file, "r") as inp:
            yaml = YAML(typ="safe")
            self._histo_defs = yaml.load(inp)

        self._output = output
        self._small = small

    @abc.abstractmethod
    def define(self, sample: model.Sample, df: DataFrame) -> dict[str, DataFrame]:
        """User method to define DataFrames.

        Args:
            sample (model.Sample): The sample to be analysed.
            df (DataFrame): The dataframe to analyse.

        Returns:
            dict of derived dataframes.
        """
        pass

    def map(self, sample: model.Sample) -> Tuple[DictHisto, DictValue]:
        """Combine the results of the child samples.

        Args:
            sample (model.Sample): The sample to be combined.

        Returns:
            A dict of histograms and a dict of counters.
        """
        if self._small:
            chain = sample.chain(SMALL_MAX_FILES)
            nr_files = min(3, len(sample))
        else:
            chain = sample.chain()
            nr_files = len(sample)

        log.debug("Sample %s has %d files.", sample, nr_files)

        df_main = ROOT.RDataFrame(chain)
        df_def = self.define(sample, df_main)
        all_values = {
            "nr_events": df_main.Count(),
        }
        all_histos: DictHisto = {}
        for histo in self._histo_defs:
            df_name = histo["dataframe"]
            try:
                df = df_def[df_name]
            except KeyError:
                log.error("Dataframe %s is not defined", df_name)
                continue
            weight = histo.get("weight", "1")
            all_values[f"{df_name}_nr_events"] = df.Count()
            all_values[f"{df_name}_sum_weights"] = df.Sum(weight)

            for h1d in histo.get("Histo1D", []):
                name = h1d["name"]
                if name in all_histos:
                    log.error("Histogram %s already defined", name)
                    continue
                title = h1d.get("title", name)

                condition = h1d.get("when")
                df1 = df if condition is None else df.Filter(condition)

                var = h1d.get("var", name)
                if not var.isidentifier():
                    df2 = df1.Define(name, var)
                    var2 = name
                else:
                    df2 = df1
                    var2 = var
                bins = h1d.get("bins")
                vbins = h1d.get("varbins")
                if bins:
                    # log.debug(
                    #     "Histo1d((%s,%s,%d,%f,%f),%s,%s)",
                    #     name,
                    #     title,
                    #     *bins,
                    #     var2,
                    #     weight,
                    # )
                    h = df2.Histo1D((name, title, *bins), var2, weight)
                elif vbins:
                    h = df2.Histo1D(
                        (name, title, len(vbins) - 1, array("d", vbins)), var2, weight
                    )
                else:
                    log.error("Histogram %s has no valid binning.")
                all_histos[name] = h

        log.debug("Definition for %s done.", sample)
        start_time = time.time()
        all_histos_result = {k: v.GetValue() for k, v in all_histos.items()}
        all_values_result = {k: v.GetValue() for k, v in all_values.items()}
        all_values_result["time"] = time.time() - start_time
        log.debug("Done for %s done.", sample)

        return all_histos_result, all_values_result

    def reduce(
        self, sample: model.SampleBase, results: list[Tuple[DictHisto, DictValue]]
    ) -> Tuple[DictHisto, DictValue]:
        """Merge the results from several samples.

        Args:
            sample (model.SampleBAse): Sample to be considered
            results (list[DictHistos]): List of result tuples of children samples

        Returns:
            Tuple[DictHisto, DictValue] ... two tuples
        """
        sum_histos: DictHisto = {}
        sum_values: DictValue = {}
        for histos, values in results:
            for k, h in histos.items():
                if k in sum_histos:
                    sum_histos[k].Add(h)
                else:
                    sum_histos[k] = h
            for k, v in values.items():
                if k in sum_values:
                    if k.startswith("min_"):
                        sum_values[k] = min(sum_values[k], v)
                    elif k.startswith("max_"):
                        sum_values[k] = max(sum_values[k], v)
                    else:
                        sum_values[k] += v
                else:
                    sum_values[k] = v

        return sum_histos, sum_values

    def gather(self, future_to_sample: dict[dd.Future, model.SampleBase]) -> None:
        """Write results and wait for all processes to finish.

        Args:
            future_to_sample (dict[dd.Future, model.SampleBase]): A mapping
                of the futures of the processes and their samples.
        """
        log.info("Writing %s", self._output.with_suffix(".root"))
        out_root = ROOT.TFile(str(self._output.with_suffix(".root")), "RECREATE")
        all_values: dict[str, DictValue] = {}
        for f in dd.as_completed(future_to_sample.keys()):
            s = future_to_sample[f]
            exc = f.exception()
            if exc is not None:
                log.error("Sample %s exception %s", str(s), exc)
                log.exception(exc)
                continue
            subdir = "_".join(s.path.parts[3:])
            ROOT.gDirectory.mkdir(subdir)
            ROOT.gDirectory.cd(subdir)
            histos, values = cast(Tuple[DictHisto, DictValue], f.result())
            all_values["/".join(s.path.parts[3:])] = values
            for h in histos.values():
                h.Write()
            ROOT.gDirectory.cd("/")

        out_root.Close()

        log.info("Writing %s", self._output.with_suffix(".json"))
        with open(self._output.with_suffix(".json"), "w") as out_json:
            json.dump(all_values, out_json)


class Processor:
    """Distribute Analysis Base Class."""

    _analysis: Analysis
    _workdir: pathlib.Path
    _cluster: dd.SpecCluster
    _client: dd.Client

    def __init__(
        self,
        root_includes: list[str],
        root_threads: int = None,
        workers: int = None,
        max_workers: int = None,
        batch: bool = None,
    ) -> None:
        """Initialse processor.

        Args:
            root_includes (list[str]): Include files for ROOT
            root_threads (int): ROOT implicit threads (0)
            workers (int): Number of worker processes (4)
            max_workers (int): Max number of workers in adaptive operation (0)
            batch (bool): Workers in batch (False)
        """
        if root_threads is None:
            root_threads = cast(
                int, _click_options.get("root_threads", cfg.sc.root_threads)
            )
        if workers is None:
            workers = cast(int, _click_options.get("workers", cfg.sc.workers))
        if max_workers is None:
            max_workers = cast(
                int, _click_options.get("max_workers", cfg.sc.max_workers)
            )
        if batch is None:
            batch = cast(bool, _click_options.get("batch", False))

        if batch and cfg.site.batch_system == "SLURM":
            log.info("Starting SLURM cluster with %d workers", workers)
            self._cluster = dask_jobqueue.SLURMCluster(
                workers,
                processes=2,
                cores=4,
                memory=cfg.site.batch_memory,
                walltime=cfg.site.batch_walltime,
                log_directory=cfg.site.log_path,
                local_directory=cfg.site.local_path,
            )
        elif batch and cfg.site.batch_system == "HTCondor":
            log.info("Starting HTCondor cluster with %d workers", workers)
            self._cluster = dask_jobqueue.HTCondorCluster(
                workers,
                processes=1,
                cores=1,
                memory=cfg.site.batch_memory,
                walltime=cfg.site.batch_walltime,
                log_directory=cfg.site.log_path,
                local_directory=cfg.site.local_path,
            )
        else:
            log.info("Starting local cluster with %d workers", workers)
            self._cluster = dd.LocalCluster(
                n_workers=workers,
                threads_per_worker=1,
                local_directory=cfg.site.local_path,
            )

        if max_workers > 0:
            log.info("Dynamic cluster with max %d workers", max_workers)
            self._cluster.adapt(maximum=cast(int, max_workers))

        self._client = dd.Client(self._cluster)
        self._client.register_worker_plugin(WorkerPlugin(root_threads, root_includes))

    def run(
        self,
        sc: cache.SamplesCache,
        period: str,
        analysis: Analysis,
        dataset: list[str],
    ) -> None:
        """Loop on samples."""

        f_to_s: dict[dd.Future, model.SampleBase] = {}
        s_to_f: dict[model.SampleBase, dd.Future] = {}
        for parent, samples, sample_groups in sc.walk(period, topdown=False):

            log.debug("Parent Sample: %s", parent)

            children: list[dd.Future] = []
            for s in samples:
                if len(s) == 0:
                    log.warning("Sample % s is empty.", s)
                    continue
                if not cache.filter_name(s.name, dataset):
                    continue
                log.debug("Submitting map for sample %s", s)
                f = self._client.submit(analysis.map, s)
                f_to_s[f] = s
                s_to_f[s] = f
                children.append(f)

            if parent.name == period:
                continue

            for g in sample_groups:
                try:
                    children.append(s_to_f[g])
                    log.debug("Adding sample group %s", g)
                except KeyError:
                    log.debug("Skipping empty sample group %s", g)

            if children:
                log.debug("Submitting reduce for sample group: %s", parent)
                f = self._client.submit(analysis.reduce, parent, children)
                f_to_s[f] = parent
                s_to_f[parent] = f

        analysis.gather(f_to_s)

    def __del__(self):

        log.warning("Client shutdown")
        self._client.shutdown()
        del self._client


def click_options():  # noqa:
    """CLI options for analysis."""

    def _set_root_threads(ctx, param, value):
        if value is not None:
            _click_options["root_threads"] = int(value)

    def _set_workers(ctx, param, value):
        if value is not None:
            _click_options["workers"] = int(value)

    def _set_max_workers(ctx, param, value):
        if value is not None:
            _click_options["max_workers"] = int(value)

    def _set_batch(ctx, param, value):
        if value is not None:
            _click_options["batch"] = bool(value)

    def _set_batch_memory(ctx, param, value):
        if value is not None:
            _click_options["batch_memory"] = value

    def _set_batch_walltime(ctx, param, value):
        if value is not None:
            _click_options["batch_walltime"] = value

    def decorator(f):
        f = click.option(
            "--root-threads",
            metavar="THREADS",
            callback=_set_root_threads,
            expose_value=False,
            default=None,
            type=click.IntRange(0),
            help="Root threads [default: 0]",
        )(f)
        f = click.option(
            "--workers",
            metavar="WORKERS",
            callback=_set_workers,
            expose_value=False,
            type=click.IntRange(1),
            help="Number of dask workers [default: 4]",
        )(f)
        f = click.option(
            "--max-workers",
            metavar="MAX",
            callback=_set_max_workers,
            expose_value=False,
            type=click.IntRange(1),
            help="Maximum dask workers in adaptive mode [default: not adaptive]",
        )(f)
        f = click.option(
            "--batch",
            is_flag=True,
            callback=_set_batch,
            expose_value=False,
            default=None,
            help="Dask workers in batch mode",
        )(f)
        f = click.option(
            "--batch-memory",
            metavar="MEMORY",
            callback=_set_batch_memory,
            expose_value=False,
            default=None,
            help="Memory for worker in batch modus [default: 2G]",
        )(f)
        f = click.option(
            "--batch-walltime",
            metavar="WALLTIME",
            callback=_set_batch_walltime,
            expose_value=False,
            default=None,
            help="Walltime for worker in batch modus [default: 01:00:00]",
        )(f)
        return f

    return decorator
