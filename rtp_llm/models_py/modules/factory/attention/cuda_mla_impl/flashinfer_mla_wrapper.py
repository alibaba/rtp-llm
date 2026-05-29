import logging
import os
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.ops import AttentionConfigs, FMHAConfig, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops

from .flashinfer_mla import (
    MlaFlashInferDecodeOp,
    MlaFlashInferPrefillOp,
    check_attention_inputs,
    warmup_flashinfer_python,
)
from .rope_emb_new import NewMlaRotaryEmbeddingOp


def _select_mla_block_id_host(attn_inputs: PyAttentionInputs) -> torch.Tensor:
    block_id = getattr(attn_inputs, "kv_cache_block_id_host", None)
    if block_id is not None and block_id.numel() > 0:
        return block_id
    return attn_inputs.kv_cache_kernel_block_id_host


class MlaFlashInferImplBase(MlaImplBase):

    def __init__(
        self,
        fmha_impl: Any,
        rope_impl: Any,
        kv_cache_write_op: MlaKVCacheWriteOp,
        attn_inputs: PyAttentionInputs,
        seq_size_per_block: int,
        attn_configs: AttentionConfigs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            attn_configs,
            attn_inputs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        warmup_flashinfer_python()
        self.seq_size_per_block = seq_size_per_block
        self.fmha_impl: Any = fmha_impl
        self.fmha_params = None
        self.rope_params = None
        self.rope_impl = rope_impl
        self.kv_cache_write_op = kv_cache_write_op
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)
        self.create_params(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs):
        if self.fmha_impl is not None:
            self.fmha_params = rtp_llm_ops.FlashInferMlaAttnParams()
            self.rope_params = self.fmha_params
            self.prepare(attn_inputs)

    def prepare(self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False):
        """Update fmha_params for prepare or CUDA Graph replay.

        Note: fmha_params is initialized in __init__, this method only updates it.
        forbid_realloc: True only when called from prepare_cuda_graph (replay); forbids buffer realloc.
        """
        assert self.fmha_impl is not None
        assert (
            self.fmha_params is not None
        ), "fmha_params should be initialized in __init__"
        check_attention_inputs(attn_inputs)
        block_id_host = _select_mla_block_id_host(attn_inputs)
        self.fmha_params.fill_params(
            attn_inputs.prefix_lengths,
            attn_inputs.sequence_lengths,
            attn_inputs.input_lengths,
            block_id_host,
            self.seq_size_per_block,
            forbid_realloc,
        )
        if forbid_realloc:
            plan_cuda_graph = getattr(self.fmha_impl, "plan_cuda_graph", None)
            if callable(plan_cuda_graph) and plan_cuda_graph(attn_inputs):
                return
        self.fmha_impl.plan(self.fmha_params)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferImplBase"
        assert self.rope_impl is not None and self.fmha_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, self.rope_params)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

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
    _debug_init_log_budget = 16

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
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            MlaFlashInferPrefillOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.v_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                weights,
                quant_config,
                attn_configs.kv_cache_dtype,
                is_cuda_graph=is_cuda_graph,
                max_batch_size=int(attn_inputs.input_lengths.size(0)),
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha,
            quant_config,
            max_seq_len,
            is_cuda_graph,
            parallelism_config,
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
        self.cuda_graph_absorb_token_limit = self.absorb_opt_len
        if is_cuda_graph:
            mtp_step_tokens = (
                int(getattr(attn_configs, "gen_num_per_cycle", 0) or 0) + 1
            )
            self.cuda_graph_absorb_token_limit = max(1, mtp_step_tokens)
        self.absorb_fmha: Optional[MlaFlashInferDecodeOp] = None
        use_absorb_fmha = (
            q_len < self.absorb_opt_len
            and (self.has_reuse_cache or is_cuda_graph)
            and attn_configs.kv_cache_dtype
            in (KvCacheDataType.BASE, KvCacheDataType.FP8)
        )
        if (
            is_cuda_graph
            and os.environ.get("RTP_LLM_DEBUG_MLA_PREFILL_PLAN", "0") != "0"
            and MlaFlashInferPrefillImpl._debug_init_log_budget > 0
        ):
            MlaFlashInferPrefillImpl._debug_init_log_budget -= 1
            logging.warning(
                "[MLA-PREFILL-IMPL] is_cuda_graph=%s q_len=%s absorb_opt_len=%s "
                "has_reuse_cache=%s kv_cache_dtype=%s use_absorb=%s input_lengths_shape=%s "
                "prefix_lengths_shape=%s cg_absorb_limit=%s",
                is_cuda_graph,
                q_len,
                self.absorb_opt_len,
                self.has_reuse_cache,
                attn_configs.kv_cache_dtype,
                use_absorb_fmha,
                tuple(attn_inputs.input_lengths.shape),
                (
                    tuple(attn_inputs.prefix_lengths.shape)
                    if attn_inputs.prefix_lengths is not None
                    else None
                ),
                self.cuda_graph_absorb_token_limit,
            )
        if use_absorb_fmha:
            max_context_len = max_seq_len
            if max_context_len <= 0 and attn_inputs.prefix_lengths is not None:
                max_context_len = int(
                    (attn_inputs.prefix_lengths + attn_inputs.input_lengths)
                    .max()
                    .item()
                )
            self.absorb_fmha = MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
                max_bs=int(attn_inputs.input_lengths.size(0)),
                max_context_len=max_context_len,
                is_cuda_graph=is_cuda_graph,
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            )
            if (not is_cuda_graph) or q_len <= self.cuda_graph_absorb_token_limit:
                self.absorb_fmha.plan(self.fmha_params)

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return attn_configs.use_mla and attn_inputs.is_prefill

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
        self.prepare(attn_inputs, forbid_realloc=True)
        if (
            self.absorb_fmha is not None
            and self._current_total_q() <= self.cuda_graph_absorb_token_limit
        ):
            self.absorb_fmha.plan_prefill_cuda_graph(self.fmha_params)

    def _current_total_q(self) -> int:
        if self.fmha_params is not None and self.fmha_params.qo_indptr_h.numel() > 0:
            return int(self.fmha_params.qo_indptr_h[-1].item())
        return 0

    def _handle_long_sequence(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ):
        """Handle long sequences using cache reuse operation."""
        # Handle cache reuse for longer sequences
        return self.fmha_impl.forward(q, compressed_kv, k_pe, kv_cache, layer_id)

    def _handle_short_sequence(
        self, q: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_id: int
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
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
    ):
        """Compute prefill context with optimized cache reuse logic."""

        total_q = self._current_total_q()
        if total_q <= 0:
            total_q = q.shape[0]

        if (
            self.absorb_fmha is not None
            and total_q <= self.cuda_graph_absorb_token_limit
        ):
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
        kv_cache: Optional[LayerKVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            topk_indices is None
        ), "topk_indices should be None for MlaFlashInferPrefillImpl"
        assert self.rope_impl is not None and self.rope_params is not None
        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, self.rope_params)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
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
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        super().__init__(
            MlaFlashInferDecodeOp(
                attn_configs.head_num,
                attn_configs.kv_lora_rank,
                attn_configs.rope_head_dim,
                attn_configs.nope_head_dim,
                attn_configs.kernel_tokens_per_block,
                attn_configs.softmax_extra_scale,
                attn_configs.use_mla,
                attn_configs.is_sparse,
                weights,
                max_bs=attn_inputs.sequence_lengths.size(0),
                max_context_len=max_seq_len,
                num_tokens=int(attn_inputs.sequence_lengths.sum().item()),
                is_cuda_graph=is_cuda_graph,
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            NewMlaRotaryEmbeddingOp(
                cos_sin_cache=cos_sin_cache,
                is_neox_style=attn_configs.rope_config.is_neox_style,
            ),
            MlaKVCacheWriteOp(
                kv_cache_dtype=attn_configs.kv_cache_dtype,
            ),
            attn_inputs,
            attn_configs.kernel_tokens_per_block,
            attn_configs,
            weights,
            cos_sin_cache,
            fmha_config,
            use_trt_fmha=False,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
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
        self.prepare(attn_inputs, forbid_realloc=True)
