"""MRTools distributed."""
import logging
from typing import Any, Callable, cast, Iterable
import pathlib
import gzip
import json

import correctionlib
import ROOT
import dask.distributed as dd
import dask_jobqueue

from mrtools import analysis, config, datasets

RDataFrame = Any
TH1 = Any
CallableDefineDF = Callable[
    [RDataFrame, str, datasets.DatasetType, str], dict[str, RDataFrame]
]

log = logging.getLogger(__name__)
cfg = config.get()


class WorkerPlugin(dd.WorkerPlugin):
    """Initialise the Dask Worker.

    Initialization of Logging, Configuration and ROOT on the
    worker processes.
    """

    cfg: config.Configuration
    log_level: int
    root_threads: int

    def __init__(self, root_threads: int = 0) -> None:
        """Init WorkerPlugin."""
        self.cfg = cfg
        self.log_level = log.getEffectiveLevel()
        self.root_threads = root_threads

    def setup(self, worker: dd.Worker) -> None:
        """Worker init."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)-8s - %(name)-16s - %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            level=self.log_level,
        )
        global cfg
        cfg = self.cfg

        ROOT.gROOT.SetBatch()
        if self.root_threads >= 0:
            ROOT.EnableImplicitMT(self.root_threads)
        correctionlib.register_pyroot_binding()


class Processor:
    _analysis: analysis.Analysis
    cluster: dd.SpecCluster
    client: dd.Client

    def __init__(
        self,
        define_dataframes: CallableDefineDF,
        histograms: list[Any],
        root_threads: int = 0,
        max_files: int = 0,
        workers: int = 0,
        max_workers: int = -1,
        batch: bool = False,
    ) -> None:
        self._analysis = analysis.Analysis(
            define_dataframes, histograms, pathlib.Path(), max_files
        )

        workers = cfg.workers if workers <= 0 else workers
        max_workers = cfg.max_workers if max_workers <= 0 else max_workers

        if batch and cfg.batch_system == "SLURM":
            log.info("Starting SLURM cluster with %d workers", workers)
            self.cluster = dask_jobqueue.SLURMCluster(
                workers,
                processes=1,
                cores=cfg.batch_cores,
                memory=cfg.batch_memory,
                walltime=cfg.batch_walltime,
                log_directory=cfg.work_path,
                local_directory=cfg.work_path,
            )
        elif batch and cfg.batch_system == "HTCondor":
            log.info("Starting HTCondor cluster with %d workers", workers)
            self.cluster = dask_jobqueue.HTCondorCluster(
                workers,
                processes=1,
                cores=cfg.batch_cores,
                memory=cfg.batch_memory,
                walltime=cfg.batch_walltime,
                log_directory=cfg.work_path,
                local_directory=cfg.work_path,
            )
        else:
            log.info("Starting local cluster with %d workers", workers)
            self.cluster = dd.LocalCluster(
                n_workers=workers,
                threads_per_worker=1,
                local_directory=cfg.work_path,
            )

        if max_workers > 0:
            log.info("Dynamic cluster with max %d workers", max_workers)
            self.cluster.adapt(max_workers=max_workers)

        self.client = dd.Client(self.cluster)
        self.client.register_worker_plugin(WorkerPlugin(root_threads))

    def run(
        self,
        the_datasets: Iterable[datasets.Dataset],
        output: pathlib.Path,
    ) -> None:
        d2f: dict[str, dd.Future] = {}
        f2d: dict[dd.Future, str] = {}
        for dataset in the_datasets:
            log.info("Dataset %s", dataset)
            if isinstance(dataset, datasets.DatasetFrom):
                log.debug("Submit map on %s", dataset)
                f = self.client.submit(
                    self._analysis.map,
                    (dataset.tree_name, [f.url() for f in dataset]),
                    dataset.name,
                    dataset.type,
                    dataset.period,
                )
                d2f[str(dataset)] = f
                f2d[f] = str(dataset)
            else:
                for parent, groups, children in dataset.walk(topdown=False):
                    f_children: list[dd.Future] = []
                    for dset in children:
                        log.debug("Submit map on %s", dset)
                        f = self.client.submit(
                            self._analysis.map,
                            (dset.tree_name, [f.url() for f in dset]),
                            dset.name,
                            dset.type,
                            dset.period,
                        )
                        d2f[str(dset)] = f
                        f2d[f] = str(dset)
                        f_children.append(f)
                    for dset_name in map(str, groups):
                        f_children.append(d2f[dset_name])
                    log.debug("Submit reduce on %s", parent)
                    f = self.client.submit(
                        self._analysis.reduce, parent.name, f_children
                    )
                    d2f[str(parent)] = f
                    f2d[f] = str(parent)

        root_file = output.with_suffix(".root")
        log.info("Writing %s ...", root_file)
        all_counters: dict[str, dict[str, int | float]] = {}
        out_root = ROOT.TFile.Open(str(root_file), "RECREATE")
        for future, result in dd.as_completed(list(d2f.values()), with_results=True):
            dset_name = f2d[cast(dd.Future, future)]
            log.info("Result from %s", dset_name)
            if result == dd.CancelledError:
                log.error("Result for %s was cancelled.")
                continue
            histos, counters = cast(
                tuple[dict[str, TH1], dict[str, int | float]], result
            )
            subdir = "_".join(dset_name.split("/")[3:])
            ROOT.gDirectory.mkdir(subdir)
            ROOT.gDirectory.cd(subdir)
            for h in histos.values():
                h.Write()
            ROOT.gDirectory.cd("/")
            all_counters[dset_name] = counters
        out_root.Close()

        json_file = output.with_suffix(".json.gz")
        log.info("Writing %s ...", json_file)
        with gzip.open(json_file, mode="wt", encoding="UTF-8") as out_json:
            json.dump(all_counters, out_json)
