"""CUDA rope modules for the py-flashinfer prefill cores.

The CUDA operators are plain Python, so the appliers subclass them directly:
callers keep the operator API they always used, while the prefill base and
its callers agree on one rope vocabulary.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    FlashinferRopeApplier,
)
from rtp_llm.models_py.modules.factory.attention.rope_applier import RopeApplier
from rtp_llm.ops import RopeStyle
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


def prefill_rope_is_fused(attn_configs: Any) -> bool:
    """Whether the configured rope must go through the fused rope operator.

    Interleaved MRoPE is only expressed by the fused rope kernels; it has no
    flashinfer rope-module equivalent, and non-interleaved MRoPE has no rope
    kernel at all (the impls reject it).
    """
    rope_config = attn_configs.rope_config
    return rope_config.style == RopeStyle.Mrope and bool(
        getattr(rope_config, "mrope_interleaved", True)
    )


def create_prefill_rope_applier(attn_configs: Any) -> Optional[RopeApplier]:
    """Rope module for the py-flashinfer prefill cores, selected by rope config.

    Mirrors the framework-side rope factories (vLLM ``get_rope``): interleaved
    MRoPE composes with the paged/ragged prefill cores through the fused rope
    operator, every other style keeps the flashinfer rope module, and a config
    without rope gets none.
    """
    if attn_configs.rope_config.style == RopeStyle.No:
        return None
    if prefill_rope_is_fused(attn_configs):
        return FusedRopeQKVOutApplier(attn_configs)
    return FlashinferRopeApplier(attn_configs)
