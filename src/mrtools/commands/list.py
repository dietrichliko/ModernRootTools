"""CLI list command for ModernROOTTools."""
import pathlib
import logging
import sys

import click

from mrtools import cache
from mrtools import utils
from mrtools import exceptions

log = logging.getLogger(".".join(__name__.split(".")[:2]))


@click.command(name="list")
@click.argument(
    "dataset-file",
    metavar="YAML",
    required=True,
    nargs=-1,
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
)
@click.option(
    "-p",
    "--period",
    metavar="PERIOD",
    default=[],
    multiple=True,
    help="Run periods. [default: all periods]",
)
@click.option(
    "--tree/--no-tree",
    default=False,
    help="List the full dataset tree",
    show_default=True,
)
def list_datasets(
    dataset_file: list[pathlib.Path],
    period: list[str],
    tree: bool,
) -> None:
    """List the datasets defined in YAML."""

    if not dataset_file:
        log.error("No dataset file given")
        sys.exit(1)

    dc = cache.DatasetCache()
    for df in dataset_file:
        dc.load(df)

    try:
        period = dc.verify_period(period)
    except exceptions.MRTError:
        log.exception("Unknown period: %s", ", ".join(period))
        sys.exit()

    for p in period:
        print(f"Period {p}:")
        if tree:
            for parent, _groups, children in dc.walk(p):
                print(
                    f"   {parent} ({parent.type.name},"
                    f"Size {utils.human_readable_size(parent.size)},"
                    f"{len(parent)} file(s))"
                )
                for child in children:
                    print(
                        f"   {child} ({child.type.name},"
                        f"Size {utils.human_readable_size(child.size)},"
                        f"{len(child)} file(s))"
                    )
        else:
            for s in dc.list(p):
                print(
                    f"   {s} ({s.type.name},"
                    f"Size {utils.human_readable_size(s.size)},"
                    f"{len(s)} file(s))"
                )
