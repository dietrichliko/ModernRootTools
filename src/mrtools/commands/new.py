"""
MRTools new command.
"""

import logging
import click

log = logging.getLogger(".".join(__name__.split(".")[:2]))


@click.command
def new():
    """
    Create new analysis skeleton.
    """
    log.fatal("Not implemented yet.")
