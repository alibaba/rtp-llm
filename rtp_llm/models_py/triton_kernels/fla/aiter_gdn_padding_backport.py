"""Verify the pinned AITER kernel's native padding-store implementation.

The new wheel includes the fix; no dependency patch is applied. Bazel packages
a source copy as provenance. Unknown/missing sources retain RTP output zeros.
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_KERNEL_SHA = "e61b398bbae88466ba3d50cf1ed807530fd3005edf21239184a77810ef7b878c"
_WRAPPER_SHA = "f130878fddb28cac46ef5be9bfca4bdf42c1d848ca44e07de1341dbc19e2f636"
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
