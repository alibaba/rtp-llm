"""Compatibility shim for optional flash-attn-3.

rtp_kernel 0.1.0 imports ``flash_attn_interface`` while importing the package,
even when FP8 sparse attention is not used.  CUDA13 does not ship a compatible
flash-attn-3 wheel yet, so keep imports working and fail only if the FP8 path is
actually called.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

_REAL_MODULE: ModuleType | None = None
_SHIM_DIR = os.path.realpath(os.path.dirname(__file__))


def _path_entry_realpath(path_entry: str) -> str:
    if not path_entry:
        path_entry = os.getcwd()
    return os.path.realpath(path_entry)


def _paths_without_shim_dir() -> list[str]:
    return [
        path_entry
        for path_entry in sys.path
        if _path_entry_realpath(path_entry) != _SHIM_DIR
    ]


def _load_real_module() -> ModuleType | None:
    global _REAL_MODULE
    if _REAL_MODULE is not None:
        return _REAL_MODULE

    original_module = sys.modules.pop(__name__, None)
    original_path = sys.path[:]
    try:
        sys.path = _paths_without_shim_dir()
        try:
            module = importlib.import_module(__name__)
        except ModuleNotFoundError as exc:
            if exc.name == __name__:
                return None
            raise

        module_file = getattr(module, "__file__", None)
        if module_file and os.path.realpath(module_file) == os.path.realpath(__file__):
            return None

        _REAL_MODULE = module
        return module
    finally:
        sys.path = original_path
        if _REAL_MODULE is None:
            if original_module is not None:
                sys.modules[__name__] = original_module
        else:
            sys.modules[__name__] = _REAL_MODULE


def _missing_flash_attn_3() -> ImportError:
    return ImportError(
        "flash_attn_interface is required only for FP8 sparse attention, but no "
        "CUDA13-compatible flash-attn-3 implementation is installed. Install a "
        "compatible flash-attn-3 wheel before using this path."
    )


def flash_attn_varlen_func(*args: Any, **kwargs: Any) -> Any:
    module = _load_real_module()
    if module is None:
        raise _missing_flash_attn_3()
    return module.flash_attn_varlen_func(*args, **kwargs)


def __getattr__(name: str) -> Any:
    module = _load_real_module()
    if module is None:
        raise AttributeError(str(_missing_flash_attn_3())) from None
    return getattr(module, name)
