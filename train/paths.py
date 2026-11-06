"""Where the training stack reads and writes. Nothing here is in the repo: set the variables, or keep the defaults
under STOPLINE_WORK.

  STOPLINE_WORK   labels/ (store.json, sessions/), meta/ (stops.json, batch*.json), runs/, hf/ (default: ./work)
  COMMA1M         comma1M's data/ folder, one folder per segment (default: $STOPLINE_WORK/comma1M)
  STOPLINE_C1M    the training frames build_c1m.py, pass_c1m.py and fast_c1m.py write (default: $STOPLINE_WORK/c1m)
  STOPLINE_SLSD   the pretraining frames prep.py writes (default: $STOPLINE_WORK/slsd)
  SLSD_RAW        the SLSDNet stop-line dataset, unzipped: img/, labels/, train.txt, test.txt (default: $STOPLINE_WORK/slsd_raw)
  STOPLINE_SITE   the labelling site (default: train/site in this repo; its frames/, manifest.json and prefill.json are
                  not committed)
  DRIVING_MODEL   the driving model's ONNX, for ego_c1m.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.abspath(os.environ.get('STOPLINE_WORK', 'work'))
COMMA1M = os.environ.get('COMMA1M', f'{WORK}/comma1M')
C1M = os.environ.get('STOPLINE_C1M', f'{WORK}/c1m')
SLSD = os.environ.get('STOPLINE_SLSD', f'{WORK}/slsd')
SLSD_RAW = os.environ.get('SLSD_RAW', f'{WORK}/slsd_raw')
SITE = os.environ.get('STOPLINE_SITE', f'{HERE}/site')
DRIVING_MODEL = os.environ.get('DRIVING_MODEL')
LABELS, META = f'{WORK}/labels', f'{WORK}/meta'
