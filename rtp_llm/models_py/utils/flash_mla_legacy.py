"""Resolve the FlashMLA runtime for existing MLA/DSv4 cache formats.

ARM retains its independently pinned legacy wheel. Prefer that package; when it
is absent, use the integrated legacy namespace. Older top-level packages require
an explicit legacy default or the legacy dense-decode native capability. V4.1
and K3 PageRR use explicit current and integrated legacy namespaces respectively.

Resolution happens per attribute access (``import_module`` consults
``sys.modules`` first), without caching a selected backend across calls.
"""

from importlib import import_module


def get_legacy_flash_mla():
    try:
        return import_module("flash_mla_legacy")
    except ModuleNotFoundError as error:
        if error.name != "flash_mla_legacy":
            raise
    try:
        return import_module("flash_mla.legacy")
    except ModuleNotFoundError as error:
        if error.name not in ("flash_mla", "flash_mla.legacy"):
            raise

    backend = import_module("flash_mla")
    default_api = getattr(backend, "DEFAULT_API", None)
    if default_api == "legacy":
        return backend
    if default_api is None:
        # CUDA12 main pins fcaae184: its top-level interface imports cuda and
        # calls cuda.dense_decode_fwd. Current-layout packages lack that native
        # entry point. This checks API capability, not a version or tensor shape.
        native = getattr(backend, "cuda", None)
        if callable(getattr(native, "dense_decode_fwd", None)):
            return backend
        # The pinned PPU wheel imports its native module in the interface,
        # rather than exposing flash_mla.cuda. Keep its original top-level API.
        interface = getattr(backend, "flash_mla_interface", None)
        native = getattr(interface, "flash_mla_cuda", None)
        if callable(getattr(native, "fwd_kvcache_mla", None)):
            return backend
    raise ImportError(
        "FlashMLA legacy cache layout requires flash_mla_legacy, flash_mla.legacy, "
        "or a verified legacy top-level API; refusing an ambiguous/current backend."
    )


def __getattr__(name):
    return getattr(get_legacy_flash_mla(), name)
