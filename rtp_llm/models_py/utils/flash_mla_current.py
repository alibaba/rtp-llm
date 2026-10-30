"""Resolve explicit current FlashMLA while preserving older wheel entry points.

New wheels expose a current namespace independently of their build default.
Older wheels retain their original top-level API; only a missing namespace may
select that API. Missing native dependencies and incomplete new wheels fail.
"""

from importlib import import_module


def get_current_flash_mla():
    try:
        return import_module("flash_mla.current")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla.current":
            raise

    backend = import_module("flash_mla")
    build_info = getattr(backend, "_build_info", None)
    if hasattr(backend, "DEFAULT_API") or hasattr(build_info, "DEFAULT_API"):
        raise ImportError(
            "FlashMLA wheel declares DEFAULT_API but lacks flash_mla.current; "
            "reinstall the complete wheel instead of using its top-level layout."
        )
    return backend


def __getattr__(name):
    return getattr(get_current_flash_mla(), name)
