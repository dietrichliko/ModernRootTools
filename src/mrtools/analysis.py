"""Modern ROOT Tools."""
import abc
import logging
import pathlib
from collections.abc import Iterable
from mrtools import configuration
from mrtools import model
from mrtools import samplescache
from typing import Any
from typing import Dict
from typing import Optional

import dask.distributed as dd
import dask_jobqueue
import ROOT

log = logging.getLogger(__name__)
config = configuration.get()

TChain = Any

class Analysis(abc.ABC):

    @abc.abstractmethod
    def map_sample(
        sample_name: str, 
        sample_type: model.SampleType, 
        sample_attrs: Dict[str, Any],
        sub_sample: int,
        df: Any,
    ) -> Tuple[List[Any], Dict[str,Any]]:
        pass

    @abc.abstactmethod
    def reduce_sample(
        sample_name: str, 
        sample_type: model.SampleType,
        histos: Dict[str, List[Any]],
        stat: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Any], Dict[str,Any]]:
        pass

    def terminate       

def map_sample(Analysis: analysis, sample_name: sample_type, sample_attrs, sub_sample, urls) ->
    

class AnalysisProcessor:
    """Distribute Analysis Base Class."""

    _analysis: Analysis
    _workdir: pathlib.Path
    _cluster: dd.SpecCluster
    _client: dd.Client
    _small: bool

    def __init__(
        self,
        analysis: Analysis,
        workdir: pathlib.Path,
        workers: int = 4,
        max_workers: int = 0,
        batch: bool = False,
        memory: str = "",
        walltime: str = "",
        small: bool = False
    ) -> None:
        self._workdir = workdir
        self._small = small

        if batch and config.site.batch == "SLURM":
            log.info("Starting SLURM cluster with %d workers", workers)
            self._cluster = dask_jobqueue.SLURMCluster(
                workers,
                processes=1,
                cores=1,
                memory=memory,
                walltime=walltime,
                log_directory=workdir,
                local_directory=workdir,
            )
        elif batch and config.site.batch == "HTCondor":
            log.info("Starting HTCondor cluster with %d workers", workers)
            self._cluster = dask_jobqueue.HTCondorCluster(
                workers,
                processes=1,
                cores=1,
                memory=memory,
                walltime=walltime,
                log_directory=workdir,
                local_directory=workdir,
            )
        else:
            log.info("Stating local cluster with %d workers", workers)
            self._cluster = dd.LocalCluster(
                n_workers=workers,
                threads_per_worker=1,
                local_directory=workdir,
            )

        if max_workers > 0:
            log.info("Dynamic cluster with max %d workers", max_workers)
            self._cluster.adapt(maximum=max_workers)

        self._client = dd.Client(self._cluster)

    def run(self, sc: samplescache.SamplesCache, period: str, recursive=False) -> None:

        if recursive:
            log.error("Recursive processing not implemented yet.")
        else:
            for sample in sc.list(period):
                if small:
                else:

                f = self._client.submit(_process_sample, sample.name, sample.type,)


Analysis
    map_sample(sample, isub)
    combine_sample(sample)
    reduce_samples(samples)
    