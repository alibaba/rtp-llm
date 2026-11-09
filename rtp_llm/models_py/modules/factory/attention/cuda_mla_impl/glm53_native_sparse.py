"""Optional SM100 paged NoPE MLA retaining the GLM FP32 KV scale format."""

import functools
import logging
import os
from pathlib import Path

import torch

_ENABLED = os.environ.get("GLM53_NATIVE_SPARSE", "1") == "1"
_INVARIANT = os.environ.get("GLM53_NATIVE_SPARSE_BATCH_INVARIANT", "0") == "1"


@functools.lru_cache(maxsize=None)
def _load_native(library):
    if not library or not Path(library).is_file():
        logging.info("GLM53 native sparse MLA unavailable; using selected-KV fallback")
        return None
    try:
        torch.ops.load_library(library)
        op = torch.ops.flash_mla_glm53_22.sparse_decode_fwd
    except (OSError, RuntimeError, AttributeError) as error:
        logging.warning(
            "GLM53 native sparse MLA unavailable; using fallback: %s", error
        )
        return None
    logging.info("GLM53 native paged sparse MLA loaded: %s", library)
    return op


def native_query_length(attn_inputs, query_rows, requests):
    """Only fixed target verification groups adjacent query rows by request."""
    if (
        attn_inputs is not None
        and getattr(attn_inputs, "is_target_verify", False)
        and requests > 0
        and query_rows > 0
        and query_rows % requests == 0
    ):
        return query_rows // requests
    return 1


class Glm53NativeSparse:
    def __init__(self):
        self.schedulers = {}
        self.native = None
        if (
            _ENABLED
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (10, 0)
        ):
            default = Path(__file__).parents[5] / "libs/flash_mla_glm53_22.abi3.so"
            self.native = _load_native(
                os.environ.get("GLM53_NATIVE_SPARSE_LIBRARY", str(default))
            )

    def supports(self, q, src):
        # Reject large Prefill/TP shapes before the caller allocates padded
        # native indices. Missing dependencies take the original path directly.
        return (
            self.native is not None
            and q.ndim == 3
            and q.is_cuda
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and q.shape[1:] == (64, 512)
            and 0 < q.shape[0] <= 384
            and src.ndim == 3
            and src.shape[1:] == (128, 528)
            and src.dtype == torch.uint8
            and src.is_contiguous()
            and src.device == q.device
        )

    def forward(self, q, src, physical_indices, scale, *, tokens_per_request=1):
        # Ordinary decode B=80 at MTP4 is 80x1, not 16x5.
        if not self.supports(q, src):
            return None
        rows, heads, width = q.shape
        if not (
            tokens_per_request > 0
            and rows % tokens_per_request == 0
            and physical_indices.shape in ((rows, 2176), (rows, 1, 2176))
            and physical_indices.dtype == torch.int32
            and physical_indices.is_contiguous()
            and physical_indices.device == q.device
        ):
            return None
        tokens = tokens_per_request
        query = q.reshape(rows // tokens, tokens, heads, width)
        indices = physical_indices.reshape(rows // tokens, tokens, -1)
        key = (rows, tokens, q.device.index, _INVARIANT)
        metadata, splits = self.schedulers.get(key, (None, None))
        # Actual kernel failures are propagated: the fallback handles explicit
        # applicability/dependency misses, not incorrect results or bad launches.
        result = self.native(
            query,
            src.unsqueeze(2),
            indices,
            None,
            None,
            metadata,
            splits,
            None,
            None,
            None,
            512,
            scale,
            _INVARIANT,
        )
        if key not in self.schedulers:
            self.schedulers[key] = (result[2], result[3])
        return result[0].reshape(rows, heads, width)
