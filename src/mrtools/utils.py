"""Modern ROOT Tools utility functions."""
import fnmatch
import logging
import zlib
from mrtools import exceptions

import click
from XRootD import client as xrd_client
from XRootD.client.flags import OpenFlags as xrd_OpenFlags


def xrd_checksum(url: str) -> int:
    """Calculate adler32 checksum of file reading its content with XRootD.

    Usage:
        checksum = xrd_checksum("root://eos.grid.vbc.at.at//eos/vbc/...")
    """
    checksum: int = 1
    with xrd_client.File() as f:
        status = f.open(url, xrd_OpenFlags.READ)
        if not status[0].ok:
            raise exceptions.MRTError(status[0].message)
        checksum = 1
        for chunk in f.readchunks():
            checksum = zlib.adler32(chunk, checksum)

    return checksum


def click_option_logging(logger, *names, **kwargs):
    """Decorator to set logging options with click."""
    if not names:
        names = ["-l", "--log-level"]

    kwargs.setdefault("metavar", "LEVEL")
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("help", "Either CRITICAL, ERROR, WARNING, INFO or DEBUG")
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("multiple", True)

    def decorator(f):
        def _set_level(ctx, param, values):
            for value in values:
                v = value.split(":")
                l = getattr(logging, v[0].upper(), None)
                if l is None:
                    raise click.BadParameter(
                        f"Must be CRITICAL, ERROR, WARNING, INFO or DEBUG, not {v}"
                    )
                if len(v) == 1:
                    logger.setLevel(l)
                else:
                    for name in logging.root.manager.loggerDict:
                        for spec in v[1:]:
                            if fnmatch.fnmatch(name, spec):
                                logging.getLogger(name).setLevel(l)

        return click.option(*names, callback=_set_level, **kwargs)(f)

    return decorator


# class LockFile:
#     """Locking on NFS.

#     Crude implementation based on os.link.

#     See https://stackoverflow.com/questions/37633951/python-locking-text-file-on-nfs
#     """

#     link_name: pathlib.Path
#     target: pathlib.Path
#     timeout: int
#     polltime: int

#     def __init__(target: PathOrStr, link_name: pathlib.Path = None, timeout:int = 300) -> None:

#         self.target = target
#         self.link_name = link_name if link_name else target.with_suffix(".lock")
#         self.timeout = timeout

#     def __enter__(self...)

#         while self.timeout > 0:
#             try:
#                 self.link_name.hardlink_to(self.target)
#                 return
#             except OSError as e:
#                 if e.errnp == errno.EEXIST:
#                     time.sleep(self.polltime)
#                     self.timeout -= self.polltime
#                 else:
#                     raise e

#         do the right thing

#     def __exit__(self):

#         self.link_name.unlink()

#     self.target.with_suffix(".lock").hardlink_to(self.target)


# def lockfile(target,link,timeout=300):
#         global lock_owner
#         poll_time=10
#         while timeout > 0:
#                 try:
#                         os.link(target,link)
#                         print("Lock acquired")
#                         lock_owner=True
#                         break
#                 except OSError as err:
#                         if err.errno == errno.EEXIST:
#                                 print("Lock unavailable. Waiting for 10 seconds...")
#                                 time.sleep(poll_time)
#                                 timeout-=poll_time
#                         else:
#                                 raise err
#         else:
#                 print("Timed out waiting for the lock.")

# def releaselock(link):
#         try:
#                 if lock_owner:
#                         os.unlink(link)
#                         print("File unlocked")
#         except OSError:
