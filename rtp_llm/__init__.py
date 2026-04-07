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


if _bootstrap_error is None and has_internal_source():
    import internal_source.rtp_llm.models_py


consume_s = time.time() - st
print(f"import in __init__ took {consume_s:.2f}s")
