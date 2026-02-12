from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType, KvCacheDataType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    check_attention_inputs,
    warmup_flashinfer_python,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import New(
    MlaRotaryEmbeddingOp, NewMlaRotaryEmbeddingParams,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops


class MlaFlashInferImplBase(MlaImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_impl: Any,
        kv_cache_write_op: MlaKVCacheWriteOp,
        attn_inputs: PyAttentionInputs,
        seq_size_per_block: int,
        is_cuda_graph: bool = False,
    ) -> None:
        warmup_flashinfer_python()
        self.seq_size_per_block = seq_size_per_block
        self.fmha_impl: Any = fmha_impl
        self.fmha_params = None
        self.rope_params = None
        self.write_cache_store_impl = None
        self.rope_impl = rope_impl
        self.kv_cache_write_op = kv_cache_write_op
        self.attn_inputs = attn_inputs
        if self.attn_inputs.is_prefill and self.attn_inputs.cache_store_inputs:
            self.write_cache_store_impl = WriteCacheStoreOp(
                self.attn_inputs.input_lengths,
                self.attn_inputs.prefix_lengths,
                self.attn_inputs.kv_cache_block_id_host,
                self.attn_inputs.cache_store_inputs,
            )
        self.create_params(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        if self.fmha_impl is not None:
            self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params = None
            if attn_inputs.is_cuda_graph is False:
                self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Update fmha_params for prepare or CUDA Graph replay.

        Note: fmha_params is initialized in __init__, this method only updates it.
        """
        assert self.fmha_impl is not None
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"
        check_attention_inputs(attn_inputs)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            attn_inputs.kv_cache_block_id_host,
            self.seq_size_per_block,
            attn_inputs.is_cuda_graph,
            attn_inputs.is_capture,
        )
        self.fmha_impl.plan(self.fmha_params)
        self.rope_params = NewMlaRotaryEmbeddingParams(self.fmha_params)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferImplBase"
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe)

        # Write compressed KV and position-encoded K to cache
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache)

        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)

        # Split query for FMHA
        q_nope, q_pe = torch.split(
            q,
            [self.fmha_impl.qk_nope_head_dim, self.fmha_impl.qk_rope_head_dim],
            dim=-1,
        )
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(q_nope, q_pe, kv_cache, layer_id)
        return res


class MlaFlashInferPrefillImpl(MlaFlashInferImplBase):
    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            MlaFlashInferPrefillOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.v_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                weights,
                quant_config,
                attn_configs.kv_cache_dtype,
            ),
            NewMlaRotaryEmbeddingOp(
                head_size=attn_configs.nope_head_dim,
                cos_sin_cache=cos_sin_cache,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
            ),
            MlaKVCacheWriteOp(
                kv_lora_rank=attn_configs.kv_lora_rank,
                rope_head_dim=attn_configs.rope_head_dim,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.tokens_per_block,
            is_cuda_graph,
        )
        self.has_reuse_cache = False
        # Type narrowing: check and assign
        if attn_inputs.prefix_lengths is not None:
            max_prefix_val = attn_inputs.prefix_lengths.max().item()  # type: ignore
            self.has_reuse_cache = max_prefix_val > 0

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        q_len = attn_inputs.input_lengths.sum().item()
        self.absorb_fmha: Optional[MlaFlashInferDecodeOp] = None
        if (
            q_len < self.absorb_opt_len
            and self.has_reuse_cache
            and attn_configs.kv_cache_dtype == KvCacheDataType.BASE
        ):
            self.absorb_fmha = MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
            )
            self.absorb_fmha.plan(self.fmha_params)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return attn_configs.use_mla and attn_inputs.is_prefill

    def _handle_long_sequence(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        """Handle long sequences using cache reuse operation."""
        # Handle cache reuse for longer sequences
        return self.fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, layer_id)

    def _handle_short_sequence(
        self, q: torch.Tensor, kv_cache: Optional[KVCache], layer_id: int
    ) -> torch.Tensor:
        """Handle short sequences using absorb operation."""
        # Split query into nope and pe components
        assert self.absorb_fmha is not None, "absorb_fmha is not initialized"
        q_nope, q_pe = torch.split(
            q,
            [self.absorb_fmha.qk_nope_head_dim, self.absorb_fmha.qk_rope_head_dim],
            dim=-1,
        )

        return self.absorb_fmha.forward(q_nope, q_pe, kv_cache, layer_id)

    def compute_prefill_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""

        if self.absorb_fmha is not None:
            return self._handle_short_sequence(q, kv_cache, layer_id)
        else:
            return self._handle_long_sequence(
                q, compressed_kv, k_pe, kv_cache, layer_id
            )

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferPrefillImpl"
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe)

        # Write compressed KV and position-encoded K to cache
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache)

        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        assert self.fmha_impl is not None
        return self.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, layer_id)


class MlaFlashInferDecodeImpl(MlaFlashInferImplBase):

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        super().__init__(
            MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
                max_bs=attn_inputs.sequence_lengths.size(0),
                max_context_len=max_seq_len,
                num_tokens=int(attn_inputs.sequence_lengths.sum().item()),
                is_cuda_graph=is_cuda_graph,
            ),
            NewMlaRotaryEmbeddingOp(
                head_size=attn_configs.nope_head_dim,
                cos_sin_cache=cos_sin_cache,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
            ),
            MlaKVCacheWriteOp(
                kv_lora_rank=attn_configs.kv_lora_rank,
                rope_head_dim=attn_configs.rope_head_dim,
                token_per_block=attn_configs.tokens_per_block,
                is_neox_style=False,
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.tokens_per_block,
            is_cuda_graph,
        )

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_configs.use_mla
            and not attn_inputs.is_prefill
            and not attn_configs.is_sparse
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs)
