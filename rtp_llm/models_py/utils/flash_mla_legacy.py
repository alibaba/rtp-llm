"""Resolve the FlashMLA runtime for existing MLA/DSv4 cache formats.

CUDA13 ships current V4.1 and legacy operators in one FlashMLA wheel. Prefer its
legacy namespace, then retain the separate legacy wheel and original top-level
API for platforms using older dependencies. V4.1 never imports this module.

Resolution happens per attribute access (``import_module`` consults
``sys.modules`` first), without caching a selected backend across calls.
"""

from importlib import import_module


def get_legacy_flash_mla():
    try:
        return import_module("flash_mla.legacy")
    except ModuleNotFoundError as error:
        if error.name not in ("flash_mla", "flash_mla.legacy"):
            raise
    try:
        return import_module("flash_mla_legacy")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla_legacy":
            raise
        return import_module("flash_mla")


def __getattr__(name):
    return getattr(get_legacy_flash_mla(), name)
