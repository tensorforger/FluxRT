import sys
import os.path as osp

_lp = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', 'LivePortrait-code'))
LIVEPORTRAIT_AVAILABLE = osp.isdir(_lp)

if LIVEPORTRAIT_AVAILABLE:
    if _lp not in sys.path:
        sys.path.insert(0, _lp)
    import src as _lp_src
    sys.modules.setdefault('liveportrait', _lp_src)
else:
    print("LivePortrait not installed, lip transfer unavailable")

from .stream_processor import *
from .utils import *
