import logging
import pathlib
import sys
from typing import cast
from typing import Any

import click

from mrtools import analysis, cache, exceptions, plotting, config

log = logging.getLogger(".".join(__name__.split(".")[:2]))
cfg = config.get()


@click.command()
@click.argument(
    "user_file",
    metavar="FILE",
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
    nargs=1,
)
@click.option("-p", "--period", default=[], multiple=True, help="Run periods.")
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
    "-n",
    "--name",
    default="{name}_{period}",
    help="Name pattern for output files",
    show_default=True,
)
@click.option(
    "--format", default=[], multiple=True, type=click.Choice(["png", "jpg", "pdf"])
)
def plot(   # noqa: C901
    user_file: pathlib.Path,
    period: list[str],
    sample_file: list[pathlib.Path],
    sample_name: list[str],
    histos_file: list[pathlib.Path],
    output: pathlib.Path | None,
    name: str,
    format: list[str],
):
    "Plot the results."
    module = analysis.load_module_from_file(user_file)
    module_dir = pathlib.Path(cast(str, module.__file__)).parent

    if not sample_file:
        sample_file = analysis.get_files_from_module(module, "DATASETS_DEF", module_dir)

    if not sample_name:
        sample_name = analysis.get_list_from_module(module, "DATASETS")

    if not histos_file:
        histos_file = analysis.get_files_from_module(module, "HISTOS_DEF", module_dir)

    histograms: list[Any] = []
    for hf in histos_file:
        histograms += analysis.load_histos(hf)

    sc = cache.DatasetCache()
    for sf in sample_file:
        sc.load(sf)

    try:
        period = sc.verify_period(period)
    except exceptions.MRTError:
        log.exception("Unknown period: %s", ", ".join(period))
        sys.exit()

    if output is None:
        output = cfg.output

    stack_plotter = plotting.StackPlotter(
        histograms,
        format,
    )

    for p in period:
        log.info("Plotting period %s", p)

        if sample_name:
            samples_iter = analysis.find_datasets(sc, p, sample_name)
        else:
            samples_iter = sc.list(p)

        stack_plotter.plot(
            samples_iter, output / name.format(name=user_file.stem, period=p)
        )
