"""One availability probe for the vendored FlashMLA wheel.

Two decode ops need it (``decode/fp8_sparse_attn_decode_op.py`` and
``fp8/decode/fp8_sparse_attn_decode_op.py``) and each carried its own copy: the
same two imports, the same ``except`` tuple, the same reasoning in the comment.
Widening that tuple therefore had to be done twice, and the next adjustment to
wheel compatibility would have had the same shape.

Availability is decided by whether the import succeeds, not by
``torch.version.cuda``. The ``>= 12.9`` requirement is a property of how the wheel
was *built*: the H20 DSV4 env loads a vendored flash_mla under torch 2.8.0+cu128
and its sm_90a cubin fine, while gating on the runtime CUDA minor version made
prefill succeed and the first decode step fail with "flash_mla wheel is required".
A genuinely incompatible wheel makes the import itself raise, which is what the
``except`` tuple is for -- including ``RuntimeError``, which is what a torch C++
extension raises on an ABI mismatch or a duplicate op registration, and which used
to escape and take the importing module down with it.
"""

import logging

_log = logging.getLogger(__name__)

FLASH_MLA_AVAILABLE = False
FLASH_MLA_IMPORT_ERROR: BaseException | None = None

try:
    from flash_mla import (
        flash_mla_with_kvcache,  # type: ignore[import-not-found]  # noqa: F401
    )
    from flash_mla import get_mla_metadata  # type: ignore[import-not-found]  # noqa: F401

    FLASH_MLA_AVAILABLE = True
except (
    ImportError,
    AttributeError,
    OSError,
    ValueError,
    RuntimeError,
) as exc:  # pragma: no cover - depends on the deployed wheel
    FLASH_MLA_IMPORT_ERROR = exc
    _log.warning(
        "[dsv4-fp8] flash_mla not available (%s); the FP8 sparse attention fast "
        "path will fail unless the reference implementation is called explicitly",
        exc,
    )


def flash_mla_available() -> bool:
    """Whether the FlashMLA entry points imported."""
    return FLASH_MLA_AVAILABLE


def flash_mla_unavailable_reason() -> str:
    """The import failure, for inclusion in an error message."""
    if FLASH_MLA_IMPORT_ERROR is None:
        return ""
    return f"{type(FLASH_MLA_IMPORT_ERROR).__name__}: {FLASH_MLA_IMPORT_ERROR}"
