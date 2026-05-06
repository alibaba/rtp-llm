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


# `from .ops import *` is needed at runtime for two reasons:
#   (a) downstream code does `from rtp_llm.ops import X` and the eager
#       import resolves the heavy `librtp_compute_ops.so` upfront.
#   (b) it AVOIDS a circular import — `rtp_llm.device.device_base`
#       imports `rtp_llm.ops.compute_ops`, which imports
#       `rtp_llm.models_py.utils.arch.is_cuda`. If `.ops` isn't
#       fully loaded BEFORE arch.py first runs, the chain
#         arch → device → device_base → compute_ops → arch (partial)
#       fails with `ImportError: cannot import name 'is_cuda' from
#       partially initialized module rtp_llm.models_py.utils.arch`
#       (run 39338093 smoke-light-sm8x tp2/beam_search_tp2 reproduction).
#
# But .ops imports torch at module level (rtp_llm/ops/__init__.py:10),
# so eager loading at PYTEST PLUGIN DISCOVERY (when pytest imports
# `rtp_llm.test.remote_tests.plugin` → triggers `rtp_llm/__init__.py`)
# pulls torch into sys.modules BEFORE conftest.py runs its xdist GPU
# slicing. Result: torch.cuInit() seen the wrong CUDA_VISIBLE_DEVICES;
# test_gpu_isolation (test_torch_not_imported_before_gpu_slice +
# test_device_count_matches_cvd) failed deterministically on ut-sm8x.
#
# Resolution: defer .ops only during pytest plugin discovery — once
# conftest.py's slicing has run (signalled by `_RTP_CONFTEST_DONE` env
# var that conftest sets at the END of its module-level code), eager
# ops loading is safe and DESIRED for downstream import correctness.
def _in_pytest_plugin_discovery() -> bool:
    """True iff this import is happening during pytest's plugin discovery.

    conftest.py sets `sys._RTP_CONFTEST_DONE = True` (Python attribute, NOT
    env var) at the end of its module-level slicing block. We use a Python-
    level flag because env vars LEAK from the controller pytest into spawned
    xdist workers — if the controller's conftest already ran (env var set),
    the worker's plugin discovery would see the inherited env var and skip
    the deferral, importing torch BEFORE the worker's own conftest runs.
    sys._RTP_CONFTEST_DONE is process-local so each xdist worker correctly
    sees False until its own conftest sets it.
    """
    if "pytest" not in sys.modules:
        return False
    if getattr(sys, "_RTP_CONFTEST_DONE", False):
        return False
    return True


try:
    import triton  # noqa: F401

    if not _in_pytest_plugin_discovery():
        from .ops import *  # noqa: F401,F403
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
