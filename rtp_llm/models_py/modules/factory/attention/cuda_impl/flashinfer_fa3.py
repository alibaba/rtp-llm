"""FlashInfer fa3 backend prefill implementation for SM90+ (Hopper).

Uses FlashInfer's BatchPrefillWithRaggedKVCacheWrapper with backend="fa3",
which runs FlashInfer's own Hopper-optimized attention kernels.
"""

import logging
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
    PyFlashinferPrefillAttnOp,
    PyFlashinferPrefillImplBase,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

_HAS_FLASHINFER_FA3 = False
try:
    from flashinfer.utils import is_sm90a_supported

    _HAS_FLASHINFER_FA3 = True
except ImportError:
    pass


class FlashInferFA3PrefillImpl(PyFlashinferPrefillImplBase):
    """FlashInfer fa3 backend prefill (SM90+ Hopper).

    Inherits from PyFlashinferPrefillImplBase but forces backend="fa3"
    instead of the default "auto" (which uses fa2).
    """

    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """Create ragged FMHA implementation with fa3 backend."""
        return PyFlashinferPrefillAttnOp(attn_configs, backend="fa3")

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        if attn_configs.rope_config.style == RopeStyle.No:
            return None
        return MhaRotaryEmbeddingOp(attn_configs)

    def _prepare_fmha_input(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """For ragged layout, reconstruct full qkv tensor from q, k, v."""
        q_flat = query.reshape(query.shape[0], -1)
        k_flat = key.reshape(key.shape[0], -1)
        v_flat = value.reshape(value.shape[0], -1)
        return torch.cat([q_flat, k_flat, v_flat], dim=-1)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASHINFER_FA3_PREFILL

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        if not _HAS_FLASHINFER_FA3:
            return False
        if not attn_inputs.is_prefill:
            return False
        if attn_configs.use_mla:
            return False
        try:
            if not is_sm90a_supported(torch.device("cuda")):
                return False
        except Exception:
            return False
        # Ragged prefill requires no prefix
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
            PyFlashinferPrefillAttnOp,
        )

        return PyFlashinferPrefillAttnOp.support(attn_inputs)
