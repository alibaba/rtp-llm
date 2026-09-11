"""Verify the pinned AITER kernel's native padding-store implementation.

The new wheel includes the fix; no dependency patch is applied. Bazel packages
a source copy as provenance. Unknown/missing sources retain RTP output zeros.
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
# Audited sources from amd-aiter 0.1.22+rocm7.2.0.git8449f41. Its decode
# kernel writes output zeros when either state index is negative.
_KERNEL_SHA = "d87523bf24a615abffac23998aa61236a8775cd906d88a0b5991cec2cbee8d38"
_WRAPPER_SHA = "0e1e9abf8e3e80affb8a8b9aa724b34fa9e776ca7cf2b2c11e48bbe9a4ed1958"
_PATCHED_SHA = _KERNEL_SHA


def _matches(path: Path, expected: str) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected


@lru_cache(maxsize=1)
def padding_safe_backend():
    """Return the native wrapper only when all three source hashes match."""
    try:
        from aiter.ops.flydsl import linear_attention_kernels as backend
        from aiter.ops.flydsl.kernels import gdr_decode

        source = Path(__file__).with_name("_aiter_gdr_decode_padding.py")
        if not (
            _matches(Path(backend.__file__), _WRAPPER_SHA)
            and _matches(Path(gdr_decode.__file__), _KERNEL_SHA)
            and _matches(source, _PATCHED_SHA)
        ):
            _LOGGER.warning(
                "GDN native padding-store source mismatch; retaining RTP zeros"
            )
            return None
        _LOGGER.info("Using verified native AITER GDN padding-store kernel: %s", source)
        return backend.flydsl_gdr_decode
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
            "GDN native padding-store verification unavailable; retaining RTP zeros: %s",
            error,
        )
        return None


def output_is_initialized_by_kernel(backend) -> bool:
    # Do not trust an arbitrary capability attribute on an external callable.
    return backend is not None and backend is padding_safe_backend()
