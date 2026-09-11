"""Use the pinned padding-store backport without rewriting installed AITER.

Bazel generates the kernel source from the checksummed AITER archive plus
patches/aiter/0001-gdr-decode-zero-padding.patch and packages it in RTP's wheel.
Unknown/missing sources retain the original backend and RTP output zeros.
"""

import hashlib
import importlib.util
import logging
import sys
import types
from functools import lru_cache
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_KERNEL_SHA = "6905ceae8fbd0e9aef664ac31cd8a07b03c4dc6c79c37234ccead26c0313d0c3"
_WRAPPER_SHA = "cc69290bb8319e693950a809ba951259f94bf16912f1cd45eed8d6e50c3f596b"
_PATCHED_SHA = "876bda2db7fe07a654e7647daaf7de3749b38e0eb73c00a671adb9b1dd0c32fe"


def _matches(path: Path, expected: str) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


@lru_cache(maxsize=1)
def padding_safe_backend():
    """Return a private wrapper only when all three source hashes match."""
    try:
        from aiter.ops.flydsl import linear_attention_kernels as backend
        from aiter.ops.flydsl.kernels import gdr_decode

        source = Path(__file__).with_name("_aiter_gdr_decode_padding.py")
        if not (
            _matches(Path(backend.__file__), _WRAPPER_SHA)
            and _matches(Path(gdr_decode.__file__), _KERNEL_SHA)
            and _matches(source, _PATCHED_SHA)
        ):
            _LOGGER.warning("GDN padding backport source mismatch; retaining RTP zeros")
            return None
        name = "aiter.ops.flydsl.kernels._rtp_gdr_decode_padding_backport"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        # The AITER package context preserves its relative tensor_shim import.
        # A distinct source/module keeps the original factory/cache untouched.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
        original = backend.flydsl_gdr_decode
        globals_copy = dict(original.__globals__)
        globals_copy["create_vk_gdr_decode_kernel"] = module.create_vk_gdr_decode_kernel
        private = types.FunctionType(
            original.__code__,
            globals_copy,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        private.__kwdefaults__ = original.__kwdefaults__
        private.__annotations__ = original.__annotations__
        _LOGGER.info("Using packaged AITER GDN padding-store backport: %s", source)
        return private
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        AttributeError,
        SyntaxError,
    ) as error:
        _LOGGER.warning(
            "GDN padding backport unavailable; retaining RTP zeros: %s", error
        )
        return None


def output_is_initialized_by_kernel(backend) -> bool:
    # Do not trust an arbitrary capability attribute on an external callable.
    return backend is not None and backend is padding_safe_backend()
