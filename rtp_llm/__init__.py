import os as _os

# Cross-repo namespace merge: extend `rtp_llm.__path__` with the sibling
# internal_source/rtp_llm tree when present (editable / monorepo dev / REAPI worker).
# Wheel install (OSS user) has no internal_source/, isdir() guard makes it no-op.
# `__path__.append` is pure path manipulation — no import side-effects, runs before
# any other rtp_llm submodule resolves.
_internal = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "..",
        "internal_source",
        "rtp_llm",
    )
)
if _os.path.isdir(_internal) and _internal not in __path__:
    __path__.append(_internal)
del _os, _internal

import os
import sys
import time
import warnings

st = time.time()
# load th_transformer.so
# Import internal models to register them
from rtp_llm.utils.import_util import has_internal_source
from rtp_llm.utils.torch_patch import *
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


_bootstrap_error = None

try:
    import triton

    from .ops import *
except Exception as exc:
    _bootstrap_error = exc
    if not _running_under_pytest():
        raise
    warnings.warn(
        f"Skipping heavy rtp_llm bootstrap during pytest startup: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )

# check triton version
# if triton.__version__ < "3.4":
#     enable_compile_monitor()


# enable_compile_monitor()


# Note: legacy `import internal_source.rtp_llm.models_py` here was dead code —
# it pointed at an empty 0-byte __init__.py with no side effect. Removed in the
# Phase-25 namespace merge; internal models register via models/internal_init.py.


consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
