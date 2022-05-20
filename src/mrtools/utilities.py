"""Modern ROOT Tools utility functions."""
import logging
import zlib
from mrtools import exceptions
from typing import cast
from typing import Union

import ROOT
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


# class LockFile:
#     """Locking on NFS.

#     Crude implementation based on os.link.

#     See https://stackoverflow.com/questions/37633951/python-locking-text-file-on-nfs
#     """

#     link_name: pathlib.Path
#     target: pathlib.Path
#     timeout: int
#     polltime: int

#     def __init__(target: PathOrStr, link_name: Optional[pathlib.Path] = None, timeout:int = 300) -> None:

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


def setAllLogLevel(level: Union[str, int]) -> None:
    """The log level of all loggers."""
    log_level = cast(int, getattr(logging, level)) if isinstance(level, str) else level
    for logger in (logging.getLogger(name) for name in logging.root.manager.loggerDict):
        logger.setLevel(log_level)


def tdr_style() -> None:
    """CMS TDR style."""
    style = ROOT.TStyle("tdrStyle", "Style for P-TDR")

    # canvas
    style.SetCanvasBorderMode(0)
    style.SetCanvasColor(ROOT.kWhite)
    style.SetCanvasDefH(600)  # Height of canvas
    style.SetCanvasDefW(600)  # Width of canvas
    style.SetCanvasDefX(0)  # Position on screen
    style.SetCanvasDefY(0)

    # pad
    style.SetPadBorderMode(0)
    # style.SetPadBorderSize(1)
    style.SetPadColor(ROOT.kWhite)
    style.SetPadGridX(False)
    style.SetPadGridY(False)
    style.SetGridColor(0)
    style.SetGridStyle(3)
    style.SetGridWidth(1)

    # For the histo:
    # style.SetHistFillColor(1)
    # style.SetHistFillStyle(0)
    style.SetHistLineColor(1)
    style.SetHistLineStyle(0)
    style.SetHistLineWidth(1)
    # style.SetLegoInnerR(Float_t rad = 0.5)
    # style.SetNumberContours(Int_t number = 20)

    style.SetEndErrorSize(2)
    # style.SetErrorMarker(20)
    style.SetErrorX(0.0)

    style.SetMarkerStyle(20)

    # For the fit/function:
    style.SetOptFit(1)
    style.SetFitFormat("5.4g")
    style.SetFuncColor(2)
    style.SetFuncStyle(1)
    style.SetFuncWidth(1)

    # For the date:
    style.SetOptDate(0)
    # style.SetDateX(Float_t x = 0.01)
    # style.SetDateY(Float_t y = 0.01)

    # For the statistics box:
    style.SetOptFile(0)
    style.SetOptStat(0)  # To display the mean and RMS:   SetOptStat("mr")
    style.SetStatColor(ROOT.kWhite)
    style.SetStatFont(42)
    style.SetStatFontSize(0.025)
    style.SetStatTextColor(1)
    style.SetStatFormat("6.4g")
    style.SetStatBorderSize(1)
    style.SetStatH(0.1)
    style.SetStatW(0.15)
    # style.SetStatStyle(1001)
    # style.SetStatX(0)
    # style.SetStatY(0)

    # Margins:
    style.SetPadTopMargin(0.05)
    style.SetPadBottomMargin(0.13)
    style.SetPadLeftMargin(0.16)
    style.SetPadRightMargin(0.02)

    # For the Global title:

    style.SetOptTitle(0)
    style.SetTitleFont(42)
    style.SetTitleColor(1)
    style.SetTitleTextColor(1)
    style.SetTitleFillColor(10)
    style.SetTitleFontSize(0.05)
    # style.SetTitleH(0) # Set the height of the title box
    # style.SetTitleW(0) # Set the width of the title box
    # style.SetTitleX(0) # Set the position of the title box
    # style.SetTitleY(0.985) # Set the position of the title box
    # style.SetTitleStyle(1001)
    # style.SetTitleBorderSize(2)

    # For the axis titles:

    style.SetTitleColor(1, "XYZ")
    style.SetTitleFont(42, "XYZ")
    style.SetTitleSize(0.06, "XYZ")
    # style.SetTitleXSize(0.02) # Another way to set the size?
    # style.SetTitleYSize(0.02)
    style.SetTitleXOffset(0.9)
    style.SetTitleYOffset(1.25)
    # style.SetTitleOffset(1.1, "Y") # Another way to set the Offset

    # For the axis labels:

    style.SetLabelColor(1, "XYZ")
    style.SetLabelFont(42, "XYZ")
    style.SetLabelOffset(0.007, "XYZ")
    style.SetLabelSize(0.05, "XYZ")

    # For the axis:

    style.SetAxisColor(1, "XYZ")
    style.SetStripDecimals(True)
    style.SetTickLength(0.03, "XYZ")
    style.SetNdivisions(510, "XYZ")
    style.SetPadTickX(1)  # To get tick marks on the opposite side of the frame
    style.SetPadTickY(1)

    # Change for log plots:
    style.SetOptLogx(0)
    style.SetOptLogy(0)
    style.SetOptLogz(0)

    # Postscript options:
    style.SetPaperSize(20.0, 20.0)
    # style.SetLineScalePS(Float_t scale = 3)
    # style.SetLineStyleString(Int_t i, const char* text)
    # style.SetHeaderPS(const char* header)
    # style.SetTitlePS(const char* pstitle)

    # style.SetBarOffset(Float_t baroff = 0.5)
    # style.SetBarWidth(Float_t barwidth = 0.5)
    # style.SetPaintTextFormat(const char* format = "g")
    # style.SetPalette(Int_t ncolors = 0, Int_t* colors = 0)
    # style.SetTimeOffset(Double_t toffset)
    # style.SetHistMinimumZero(kTRUE)

    style.cd()

    return style
