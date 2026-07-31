import os
from contextlib import nullcontext
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
        if (
            self.fmha_impl is not None
            and os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        ):
            input_host = getattr(attn_inputs, "input_lengths_host", None)
            prefix_host = getattr(attn_inputs, "prefix_lengths_host", None)
            if input_host is not None and input_host.numel():
                input_values = [int(value) for value in input_host.tolist()]
                prefix_values = (
                    [int(value) for value in prefix_host.tolist()]
                    if prefix_host is not None and prefix_host.numel()
                    else [0] * len(input_values)
                )
                self.fmha_impl.total_kv_lens_hint = sum(input_values) + sum(
                    prefix_values
                )
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
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        prefix_lengths = (
            getattr(attn_inputs, "prefix_lengths_host", None)
            if use_host_metadata
            else None
        )
        sequence_lengths = (
            getattr(attn_inputs, "sequence_lengths_host", None)
            if use_host_metadata
            else None
        )
        input_lengths = (
            getattr(attn_inputs, "input_lengths_host", None)
            if use_host_metadata
            else None
        )
        prefix_lengths = (
            prefix_lengths
            if prefix_lengths is not None and prefix_lengths.numel()
            else attn_inputs.prefix_lengths
        )
        sequence_lengths = (
            sequence_lengths
            if sequence_lengths is not None and sequence_lengths.numel()
            else attn_inputs.sequence_lengths
        )
        input_lengths = (
            input_lengths
            if input_lengths is not None and input_lengths.numel()
            else attn_inputs.input_lengths
        )
        self.fmha_params.fill_params(
            prefix_lengths,
            sequence_lengths,
            input_lengths,
            attn_inputs.kv_cache_kernel_block_id_host,
            self.seq_size_per_block,
            forbid_realloc,
        )
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

        def profile(stage: str, shape: tuple[int, ...]):
            if os.environ.get("KIMI_K3_PERF_MODE", "0").strip() != "1":
                return nullcontext()
            shape_text = "x".join(str(dim) for dim in shape)
            return torch.autograd.profiler.record_function(
                f"layer.{layer_id}.mla.{stage}[shape={shape_text}]"
            )

        q_pe = q[:, :, self.fmha_impl.qk_nope_head_dim :]

        # Apply RoPE to Q and K
        with profile("nope_rope_adapter", tuple(q_pe.shape)):
            self.rope_impl.forward(q_pe, k_pe, self.rope_params)

        # Write compressed KV and position-encoded K to cache
        with profile(
            "kv_cache_update_normalized_latent_plus_suffix",
            tuple(compressed_kv.shape),
        ):
            self.kv_cache_write_op.forward(
                compressed_kv, k_pe, kv_cache, self.rope_params
            )

        with profile("cache_store_publish_noop_for_standalone_prefill", ()):
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
        with profile("flashinfer_causal_attention_context", tuple(q.shape)):
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
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        prefix_host = (
            getattr(attn_inputs, "prefix_lengths_host", None)
            if use_host_metadata
            else None
        )
        if prefix_host is not None and prefix_host.numel():
            self.has_reuse_cache = max(int(v) for v in prefix_host.tolist()) > 0
        elif attn_inputs.prefix_lengths is not None:
            # Compatibility fallback for older bindings.
            self.has_reuse_cache = bool(attn_inputs.prefix_lengths.max().item() > 0)

        self.absorb_opt_len = (
            fmha_config.absorb_opt_len if fmha_config is not None else 1024
        )
        input_host = (
            getattr(attn_inputs, "input_lengths_host", None)
            if use_host_metadata
            else None
        )
        q_len = (
            sum(int(v) for v in input_host.tolist())
            if input_host is not None and input_host.numel()
            else attn_inputs.input_lengths.sum().item()
        )
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
                attn_configs.kernel_tokens_per_block,
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
        use_host_metadata = os.environ.get("KIMI_K3_USE_HOST_METADATA", "0") == "1"
        sequence_lengths_host = (
            getattr(attn_inputs, "sequence_lengths_host", None)
            if use_host_metadata
            else None
        )
        if sequence_lengths_host is not None and sequence_lengths_host.numel() > 0:
            sequence_values = [int(value) for value in sequence_lengths_host.tolist()]
            max_bs = len(sequence_values)
            num_tokens = sum(sequence_values)
        else:
            max_bs = attn_inputs.sequence_lengths.size(0)
            num_tokens = int(attn_inputs.sequence_lengths.sum().item())
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
                max_bs=max_bs,
                max_context_len=max_seq_len,
                num_tokens=num_tokens,
                is_cuda_graph=is_cuda_graph,
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
