"""CUDA applier for the fused rope/KV-cache operator.

The CUDA operator is plain Python, so the applier subclasses it directly:
callers keep the operator API they always used, while the prefill base and
the MRoPE implementation agree on one rope vocabulary.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.rope_applier import RopeApplier
from rtp_llm.ops.compute_ops import FusedRopeKVCachePrefillOpQKVOut


class FusedRopeQKVOutApplier(FusedRopeKVCachePrefillOpQKVOut, RopeApplier):
    """Packed QKV in, packed QKV out; stores K/V when handed a cache."""

    fused_kv_write = True
    owns_params = True

    def apply(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[Any] = None,
        params: Optional[Any] = None,
    ) -> torch.Tensor:
        return self.forward(qkv, kv_cache, params)
