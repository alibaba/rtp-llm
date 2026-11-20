"""Resolve the FlashMLA runtime for existing MLA/DSv4 cache formats.

The CUDA 13 wheels expose legacy kernels either as a separate distribution or
as the flash_mla.legacy subpackage. Older wheels expose them at the top level.

Resolution happens per attribute access so tests can substitute a backend in
``sys.modules`` and restore it without leaving a stale reference here.
"""

from importlib import import_module


def get_legacy_flash_mla():
    for name in ("flash_mla_legacy", "flash_mla.legacy"):
        try:
            return import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
    return import_module("flash_mla")


def __getattr__(name):
    return getattr(get_legacy_flash_mla(), name)
