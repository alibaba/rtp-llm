import importlib
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# DeepGEMM wheel shadow (DSV4_SM120, env-gated, default off).
#
# The runfiles bundle a deep_gemm wheel whose C++ asserts "Unsupported
# architecture" on SM120 for the hyperconnection / split-k kernels the MHC
# pre-norm path uses (surface symptom: "TileLang mhc_pre failed" wrapping it).
# The PD launcher sidesteps this via PYTHONPATH, which the bazel test wrapper
# cannot honour: it appends an inherited PYTHONPATH LAST and de-duplicates
# keeping the FIRST occurrence, so the bundled wheel always wins.
#
# Set DSV4_DEEPGEMM_SHADOW_PATH to a site-packages-like directory - e.g. the
# SM120-capable nv_dev build in DeepGEMM/build/lib.linux-x86_64-cpython-310 -
# and it is prepended to sys.path here, before any module imports deep_gemm.
# ---------------------------------------------------------------------------
_shadow = os.environ.get("DSV4_DEEPGEMM_SHADOW_PATH")
if _shadow and _shadow not in sys.path:
    sys.path.insert(0, _shadow)


def __getattr__(name: str) -> Any:
    """Preserve old top-level access without importing C++ ops eagerly."""
    if name == "_ft_pickler":
        module = importlib.import_module("rtp_llm._ft_pickler")
        globals()[name] = module
        return module

    if name == "enable_compile_monitor":
        from rtp_llm.utils.triton_compile_patch import enable_compile_monitor

        globals()[name] = enable_compile_monitor
        return enable_compile_monitor

    ops = importlib.import_module("rtp_llm.ops")
    try:
        value = getattr(ops, name)
    except AttributeError as e:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from e
    globals()[name] = value
    return value
