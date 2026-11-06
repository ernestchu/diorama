"""
Diorama in a Linux browser, for development: a recorded drive run through the
driving model and the stop-line model into a file the Diorama page plays beside
the drive's footage. The iPhone app does not use any of this.

The page (web/page) and this package come from jetlink's Diorama (MIT). The
drive is read and the driving model run with commasim's code (sim/commasim).
"""
import sys
from pathlib import Path

_SIM = str(Path(__file__).resolve().parent.parent.parent / 'sim')
if _SIM not in sys.path:
  sys.path.insert(0, _SIM)
