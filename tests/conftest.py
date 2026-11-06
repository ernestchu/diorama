import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'sim'), str(ROOT / 'web')]  # diorama, commasim's reference parse and warp, dioramaweb
