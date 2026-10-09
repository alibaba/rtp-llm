"""Resolve the FlashMLA runtime for existing MLA/DSv4 cache formats.

CUDA13 ARM ships separate current V4.1 and legacy wheels. Other platform locks
retain their original FlashMLA distribution, so use that only when the isolated
legacy distribution is absent. V4.1 never imports this module.

Resolution happens per attribute access (``import_module`` consults
``sys.modules`` first), so tests that substitute a fake module under either
name intercept these imports, and restoring ``sys.modules`` restores the real
backend without stale caching here.
"""

from importlib import import_module


def get_legacy_flash_mla():
    try:
        return import_module("flash_mla_legacy")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla_legacy":
            raise
        return import_module("flash_mla")


def __getattr__(name):
    return getattr(get_legacy_flash_mla(), name)
