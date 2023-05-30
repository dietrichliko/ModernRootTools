"""CLI list command for ModernROOTTools."""
import pathlib
import logging
import sys

import click

from mrtools import cache
from mrtools import utils

log = logging.getLogger(".".join(__name__.split(".")[:2]))


@click.command(name="list")
@click.option(
    "-f",
    "--sample-file",
    default=[],
    multiple=True,
    type=click.Path(dir_okay=False, exists=True, readable=True, path_type=pathlib.Path),
)
@click.option("-p", "--period", default=[], multiple=True, help="Run periods.")
@click.option("--tree/--no-tree", default=False, help="List the full sample tree")
def list_samples(
    period: list[str],
    sample_file: list[pathlib.Path],
    tree: bool,
) -> None:
    """List the datasets defined in the sample cache."""

    if not sample_file:
        log.error("No sample file given")
        sys.exit(1)

    sc = cache.DatasetCache()
    for sf in sample_file:
        sc.load(sf)

    period = sc.verify_period(period)

    for p in period:
        print(f"Period {p}:")
        if tree:
            for parent, _groups, children in sc.walk(p):
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
            for s in sc.list(p):
                print(
                    f"   {s} ({s.type.name},"
                    f"Size {utils.human_readable_size(s.size)},"
                    f"{len(s)} file(s))"
                )
