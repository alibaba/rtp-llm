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
from rtp_llm.utils.triton_compile_patch import enable_compile_monitor


# torch_patch monkey-patches torch.concat for FP8 fallback. Apply eagerly
# only if torch is already in sys.modules (typical runtime where the caller
# imported torch before rtp_llm). When rtp_llm is loaded via pytest entry-
# point plugin discovery (e.g. remote-gpu / rtp-ci-profile), torch has NOT
# been imported yet — applying the patch would force `import torch` BEFORE
# conftest.py GPU slicing runs, breaking test_gpu_isolation
# (test_torch_not_imported_before_gpu_slice + test_device_count_matches_cvd).
#
# For the deferred path (production / model-load-after-rtp_llm-import), a
# sys.meta_path hook applies the patch the moment torch is first imported,
# so the FP8 concat fallback is always installed before any model code runs.
def _apply_torch_patch() -> None:
    # Importing torch_patch as a module triggers its top-level
    # `torch.concat = custom_concat` monkey-patch as a side-effect; we don't
    # need any names from it. `from ... import *` is invalid inside a
    # function (Python SyntaxError), so we just import the module.
    import rtp_llm.utils.torch_patch  # noqa: F401


if "torch" in sys.modules:
    _apply_torch_patch()
else:
    import importlib.machinery as _machinery
    from importlib.abc import MetaPathFinder as _MetaPathFinder

    class _DeferredTorchPatchFinder(_MetaPathFinder):
        """Apply torch_patch the moment torch is first imported."""

        def find_spec(self, name, path=None, target=None):
            if name != "torch":
                return None
            # De-register first to avoid recursion when PathFinder loads torch.
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = _machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            _orig_exec_module = spec.loader.exec_module

            def _exec_then_patch(module):
                _orig_exec_module(module)
                _apply_torch_patch()

            spec.loader.exec_module = _exec_then_patch  # type: ignore[assignment]
            return spec

    sys.meta_path.insert(0, _DeferredTorchPatchFinder())
    # Keep _machinery / _MetaPathFinder / _DeferredTorchPatchFinder bound at
    # module level — find_spec is a closure that resolves _machinery when torch
    # is imported, which may be much later than rtp_llm.__init__ ran.


def _running_under_pytest() -> bool:
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


_bootstrap_error = None

# `from .ops import *` previously ran here, but rtp_llm.ops imports torch
# at module level (rtp_llm/ops/__init__.py:10). Eager loading meant pytest
# entry-point plugin discovery (which imports `rtp_llm.test.remote_tests.
# plugin` → triggers `rtp_llm/__init__.py`) pulled torch into sys.modules
# BEFORE conftest.py's GPU slicing had a chance to set CUDA_VISIBLE_DEVICES.
# Result: cuInit() saw the wrong CVD; test_gpu_isolation
# (test_torch_not_imported_before_gpu_slice + test_device_count_matches_cvd)
# failed deterministically on every ut-sm8x run.
#
# Defer the ops import: downstream modules use `from rtp_llm.ops import X`
# explicitly (start_server.py, models/llama.py, pipeline.py …), which Python
# resolves on demand without needing the eager star-import. Keep `import
# triton` here so the `_bootstrap_error` still surfaces missing triton at
# rtp_llm import time (triton itself does NOT pull torch).
try:
    import triton  # noqa: F401
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
