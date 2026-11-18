import importlib
import os as _os
from typing import Any

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

import sys

from rtp_llm.utils.import_util import has_internal_source


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


def __getattr__(name: str) -> Any:
    """Preserve old top-level access without importing C++ ops eagerly."""
    # Test discovery probes package hooks even when collecting pure Python tests.
    if name in {
        "pytest_plugins",
        "setUpModule",
        "tearDownModule",
        "setup_module",
        "teardown_module",
    } or (name.startswith("__") and name.endswith("__")):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
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
