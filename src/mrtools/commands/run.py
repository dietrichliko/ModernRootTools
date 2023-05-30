"""Cli com and run for ModernROOTTools."""

import pathlib
from typing import cast
from typing import Sequence
from typing import Any
import logging
import sys

import click


from mrtools import analysis
from mrtools import cache
from mrtools import exceptions
from mrtools import config

log = logging.getLogger(".".join(__name__.split(".")[:2]))
cfg = config.get()


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.argument(
    "user_file",
    metavar="FILE",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
    nargs=1,
)
@click.argument("extra", nargs=-1, type=click.UNPROCESSED)
@click.option(
    "-p",
    "--period",
    default=[],
    multiple=True,
    help="Run periods [default: all periods]",
)
@click.option(
    "-f",
    "--sample-file",
    default=[],
    multiple=True,
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
)
@click.option(
    "-s",
    "--sample",
    "sample_name",
    metavar="Name",
    default=[],
    multiple=True,
    help="Sample name",
)
@click.option(
    "-h",
    "--histos-file",
    default=[],
    multiple=True,
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
    help="Output directory [default from config]",
)
@click.option(
    "-n", "--name", default="{name}_{period}", help="Name pattern for output files"
)
@click.option(
    "--root-threads",
    default=0,
    type=click.IntRange(0),
    help="Limit ROOT Multithreading",
    show_default=True,
)
@click.option(
    "--root-mt/--no-root-mt",
    default=True,
    help="ROOT multithreading",
    show_default=True,
)
@click.option(
    "--max-files", default=0, help="Max number of files per sample to process."
)
def run(
    user_file: pathlib.Path,
    extra: Sequence[str],
    period: list[str],
    sample_file: list[pathlib.Path],
    sample_name: list[str],
    histos_file: list[pathlib.Path],
    output: pathlib.Path | None,
    name: str,
    root_threads: int,
    root_mt: bool,
    max_files: int,
) -> None:
    """Run the analysis."""
    module = analysis.load_module_from_file(user_file)
    module_dir = pathlib.Path(cast(str, module.__file__)).parent

    if not sample_file:
        sample_file = analysis.get_files_from_module(module, "DATASETS_DEF", module_dir)

    if not sample_name:
        sample_name = analysis.get_list_from_module(module, "DATASETS")

    if not histos_file:
        histos_file = analysis.get_files_from_module(module, "HISTOS_DEF", module_dir)

    if output is None:
        output = cfg.output

    dc = cache.DatasetCache()
    for sf in sample_file:
        dc.load(sf)

    try:
        period = dc.verify_period(period)
    except exceptions.MRTError:
        log.exception("Unknown period: %s", ", ".join(period))
        sys.exit()

    analyzer = module.get_analysis.main(
        args=extra, prog_name="analyzer", standalone_mode=False
    )

    histograms: list[Any] = []
    for hf in histos_file:
        histograms += analysis.load_histos(hf)

    rt = root_threads if root_mt else -1
    proc = analysis.Processor(analyzer, histograms, rt, max_files)
    for p in period:
        log.info("Analyzing period %s", p)
        if sample_name:
            samples_iter = analysis.find_samples(dc, p, sample_name)
        else:
            samples_iter = dc.list(p)

        proc.run(
            samples_iter,
            output / name.format(name=user_file.stem, period=p),
        )
