# load th_transformer.so
from .ops import *
from rtp_llm.utils.torch_patch import *

# Import internal models to register them
try:
    import internal_source.rtp_llm.models_py
except ImportError:
    logging.warning("Failed to import internal_source models")

import os
import pdb
import sys

from rtp_llm.config.log_config import LOGGING_CONFIG
from rtp_llm.utils.torch_patch import *


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
