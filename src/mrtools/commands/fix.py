"""
MRTools fix data command.
"""
import logging

import click

log = logging.getLogger(".".join(__name__.split(".")[:2]))


@click.command
def fix():
    """
    Fix data for HLT schema.
    """
    log.fatal("Not implemented yet.")
