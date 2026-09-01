import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    fp32_state_pool_view,
    fp8_pool_view,
)
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import KVCache, KVCacheRegionName
from rtp_llm.utils.model_weight import W

_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
        fused_logits_head_gate,
    )
else:
    CudaFp8GEMMLinear = None  # type: ignore
    fused_logits_head_gate = None  # type: ignore


def _project_with_optional_fp8(
    projection: nn.Module,
    bf16_input: torch.Tensor,
    fp8_input: Optional[torch.Tensor],
    input_scale: Optional[torch.Tensor],
) -> torch.Tensor:
    if (
        fp8_input is not None
        and input_scale is not None
        and CudaFp8GEMMLinear is not None
        and isinstance(projection, CudaFp8GEMMLinear)
    ):
        return projection(fp8_input, input_scales=input_scale)
    return projection(bf16_input)


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        scale_fmt: Optional[str] = "none",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        # Resolve once at init: HWKernelConfig.enable_fuse_kernels (or env
        # ``ENABLE_FUSE_KERNELS``) → ``self._fuse_logits_head_gate``. Keep it
        # out of the forward path so it's free at decode (no env / config
        # lookup per token).
        self._fuse_logits_head_gate = (
            fuse_kernels_enabled(hw_kernel_config)
            and fused_logits_head_gate is not None
        )

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk
        self.compress_ratio = int(getattr(attn_config, "indexer_compress_ratio", 1))
        self._block_table_group_ids: Optional[Dict[int, int]] = None

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        # Owner (physical) block size used by the C++ KVCacheAllocator / CPSlotMapper
        # to decide page-RR ownership. bpk = owner_tpb / kernel_tpb >= 1. Mirrors the
        # DSV4 indexer's _kv_owner_tokens_per_block contract; threaded into
        # _get_topk_ragged_cp so build_indexer_cp_chunk_plan computes per-rank padded
        # local KV lens and restore indices at the owner granularity that matches
        # how prefill writes were laid out via cp_params.sharded_slot_mapping.
        kernel_tpb = int(attn_config.kernel_tokens_per_block)
        owner_tpb = int(getattr(attn_config, "tokens_per_block", kernel_tpb))
        if owner_tpb <= 0 or kernel_tpb <= 0 or owner_tpb % kernel_tpb != 0:
            owner_tpb = kernel_tpb
        self._kv_owner_tokens_per_block = owner_tpb
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        if self.compress_ratio > 1:
            if self.compress_ratio != 4:
                raise ValueError(
                    "GLM-5.3-Flash compressed indexer requires ratio 4, got "
                    f"{self.compress_ratio}"
                )
            from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8

            # RTP loader linear tensors use [in, out]. CompressorFP8 consumes
            # native F.linear weights [out, in], while the KPool gate is loaded
            # directly from the checkpoint and already has [out, in] layout.
            k_weight = weights[W.mla_indexer_k_w].T.contiguous()
            gate_weight = weights[W.v4_indexer_compressor_wgate].contiguous()
            weights_projection = weights[W.mla_indexer_weights_proj_w].T.contiguous()
            self.compressed_indexer = IndexerFP8(
                dim=int(k_weight.shape[1]),
                q_lora_rank=int(weights[W.mla_indexer_qb_w].shape[0]),
                index_n_heads=self.index_n_heads,
                index_head_dim=self.index_head_dim,
                rope_head_dim=self.rope_head_dim,
                index_topk=self.index_topk,
                compress_ratio=self.compress_ratio,
                max_batch_size=1024,
                max_seq_len=int(attn_config.max_seq_len),
                norm_eps=layernorm_eps,
                q_projection=self.wq_b,
                weights_projection=weights_projection,
                compressor_weights={
                    "ape": weights[W.v4_indexer_compressor_ape],
                    "wkv": k_weight,
                    "wgate": gate_weight,
                    # KPool applies LayerNorm before pooling, so the generic
                    # post-pool RMS weight is intentionally unused.
                    "norm": torch.ones_like(weights[W.mla_indexer_k_norm_w]),
                },
                compressor_kpool_mode=True,
                compressor_pre_norm_weight=weights[W.mla_indexer_k_norm_w],
                compressor_pre_norm_bias=weights[W.mla_indexer_k_norm_b],
                rotate_q=True,
            )
            return

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # Pre-contiguify weight for the fused Triton kernel (one-time init
        # copy). Production weight is often a transposed view [N, K] of
        # underlying [K, N] storage; the small-T per-(t,n) kernel needs
        # contiguous weight for coalesced 1D loads. Use plain attribute
        # reassignment (not `.data = ...`) — the latter does an in-place
        # `set_` of the underlying storage that leaves the tensor in a
        # state where `F.linear` under cuda-graph capture + inference_mode
        # trips PyTorch's version-counter check.
        if (
            self._fuse_logits_head_gate
            and hasattr(self.weights_proj, "weight")
            and not self.weights_proj.weight.is_contiguous()
        ):
            self.weights_proj.weight = self.weights_proj.weight.contiguous()
        self.cos_sin_cache = global_weights[W.rope_cos_sin_cache]

        self.indexer_op = IndexerOp(
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            rope_head_dim=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            blocksize=self.blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=self.is_neox_style,
        )

    def _bind_compressed_pools(
        self, global_kv_cache: KVCache, attention_inputs: Any
    ) -> tuple[torch.Tensor, int]:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            INDEXER_KV,
            INDEXER_STATE,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
            require_pool_tokens_per_block,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            build_block_tables_batched_from_group_ids,
        )

        if self._block_table_group_ids is None:
            raise RuntimeError(
                "GLM-5.3-Flash compressed indexer block-table group ids "
                "were not bound during model initialization"
            )
        block_tables = build_block_tables_batched_from_group_ids(
            self._block_table_group_ids, attention_inputs
        )
        if block_tables is None:
            raise RuntimeError(
                "GLM-5.3-Flash compressed indexer block tables are unavailable"
            )
        kv_base = global_kv_cache.get_raw_pool_tensor(
            self.layer_idx, KVCacheRegionName.INDEXER_KV
        )
        state_base = global_kv_cache.get_raw_pool_tensor(
            self.layer_idx, KVCacheRegionName.INDEXER_STATE
        )
        kv_view, kv_eb = fp8_pool_view(kv_base, entry_bytes=132)
        state_view, state_eb = fp32_state_pool_view(
            state_base, state_width=2 * self.index_head_dim
        )
        kv_key = int(INDEXER_KV)
        state_key = int(INDEXER_STATE)
        kv_block_table = block_tables[kv_key]
        self.compressed_indexer.set_pool_context(
            kv_view,
            kv_block_table,
            kv_eb,
            state_view,
            block_tables[state_key],
            state_eb,
            state_tokens_per_block=require_pool_tokens_per_block(
                global_kv_cache, region=state_key
            ),
            kv_tokens_per_block=require_pool_tokens_per_block(
                global_kv_cache, region=kv_key
            ),
            kv_owner_tokens_per_block=int(global_kv_cache.seq_size_per_block),
        )
        return kv_block_table, kv_eb

    @staticmethod
    def _has_prefix(attention_inputs: Any) -> bool:
        prefix_host = getattr(attention_inputs, "prefix_lengths_host", None)
        if prefix_host is not None and prefix_host.numel():
            return any(int(value) > 0 for value in prefix_host.tolist())
        prefix = attention_inputs.prefix_lengths
        return bool(prefix.numel() and prefix.max().item() > 0)

    def _forward_compressed(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        fmha_params: Any,
        attention_inputs: Any,
        global_kv_cache: KVCache,
        use_fast_path: bool,
        cp_params: Any,
    ) -> Optional[torch.Tensor]:
        kv_block_table, kv_eb = self._bind_compressed_pools(
            global_kv_cache, attention_inputs
        )
        is_multi_token_decode = bool(
            getattr(attention_inputs, "is_target_verify", False)
        ) or bool(getattr(attention_inputs, "is_draft_extend", False))
        is_regular_prefill = bool(attention_inputs.is_prefill) and not (
            is_multi_token_decode
        )
        cp_ctx = None
        workspace = None
        try:
            if is_regular_prefill:
                position_ids = fmha_params.positions_d
                req_id_per_token = fmha_params.batch_indice_d
                input_lengths = attention_inputs.input_lengths
                prefix_lengths = attention_inputs.prefix_lengths
                cu_seqlens = attention_inputs.cu_seqlens
                batch_size = int(input_lengths.numel())

                if self._prefill_cp_enabled():
                    if cp_params is None:
                        raise RuntimeError(
                            "GLM-5.3-Flash compressed prefill CP requires cp_params"
                        )
                    cp_info = getattr(
                        attention_inputs, "context_parallel_info", None
                    )
                    if cp_info is None:
                        raise RuntimeError(
                            "GLM-5.3-Flash compressed prefill CP requires "
                            "context_parallel_info"
                        )
                    from rtp_llm.models_py.modules.dsv4.cp import build_cp_context
                    from rtp_llm.models_py.modules.dsv4.prefill_workspace import (
                        PrefillWorkspace,
                    )

                    cp_size = int(getattr(cp_params, "cp_size", 0))
                    cp_rank = int(getattr(cp_params, "cp_rank", -1))
                    if cp_size <= 1 or cp_rank < 0 or cp_rank >= cp_size:
                        raise RuntimeError(
                            "invalid GLM-5.3-Flash compressed prefill CP geometry: "
                            f"cp_size={cp_size} cp_rank={cp_rank}"
                        )
                    cp_ctx = build_cp_context(
                        cp_info,
                        cp_size,
                        cp_rank,
                        int(hidden_states.shape[0]),
                        hidden_states.device,
                        position_offset=prefix_lengths,
                        kv_cache_sharded=bool(
                            getattr(cp_params, "kv_cache_sharded", False)
                        ),
                    )
                    if cp_ctx.input_lengths_global is None:
                        raise RuntimeError(
                            "GLM-5.3-Flash compressed prefill CP requires global "
                            "input lengths"
                        )
                    if cp_ctx.req_id_per_token is None:
                        raise RuntimeError(
                            "GLM-5.3-Flash compressed prefill CP requires per-token "
                            "request ids"
                        )

                    # IndexerFP8's nested KPool compressor gathers the rank-local
                    # projections, restores global request order, and then writes
                    # only the page-RR entries owned by this CP rank.  Its CP
                    # gather/restore implementation deliberately requires an
                    # explicit workspace; the KPool projection width is 2*128.
                    workspace = PrefillWorkspace(
                        hidden_states.device,
                        q_rows=0,
                        q_dim=0,
                        reserve_cp=True,
                        cp_rows=cp_ctx.padded_seq_len,
                        main_w=0,
                        idx_w=2 * self.index_head_dim,
                        swa_w=0,
                        align_bytes=256,
                    )
                    self.compressed_indexer.set_cp_ctx(cp_ctx)
                    self.compressed_indexer.compressor.set_cp_ctx(cp_ctx)

                    position_ids = cp_ctx.global_positions
                    req_id_per_token = cp_ctx.req_id_per_token
                    input_lengths = cp_ctx.input_lengths_global
                    prefix_lengths = cp_ctx.prefix_lengths
                    batch_size = int(input_lengths.numel())
                    local_lengths = torch.tensor(
                        cp_ctx.chunk_lengths_per_req,
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    cu_seqlens = torch.zeros(
                        batch_size + 1,
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    cu_seqlens[1:] = torch.cumsum(local_lengths, dim=0)

                meta = self.compressed_indexer.prepare(
                    bsz=1,
                    seqlen=int(hidden_states.shape[0]),
                    sp_int=0,
                    device=hidden_states.device,
                    kv_block_table=kv_block_table,
                    kv_eb=kv_eb,
                    use_varlen=True,
                    batch_size=batch_size,
                    cu_seqlens=cu_seqlens,
                    input_lengths=input_lengths,
                    prefix_lengths=prefix_lengths,
                    position_ids=position_ids,
                    req_id_per_token=req_id_per_token,
                    has_prefix=self._has_prefix(attention_inputs),
                )
                topk = self.compressed_indexer(
                    hidden_states,
                    q_lora,
                    meta,
                    workspace=workspace,
                )
                if cp_ctx is not None:
                    total_local_ids = getattr(cp_params, "total_local_ids", None)
                    if total_local_ids is None:
                        raise RuntimeError(
                            "GLM-5.3-Flash compressed prefill CP requires "
                            "total_local_ids"
                        )
                    topk = topk.index_select(
                        0, total_local_ids.to(device=topk.device, dtype=torch.long)
                    )
            else:
                batch_size = int(attention_inputs.input_lengths.numel())
                total_tokens = int(hidden_states.shape[0])
                if batch_size <= 0 or total_tokens % batch_size:
                    raise RuntimeError(
                        "GLM-5.3-Flash compressed decode requires a uniform "
                        f"query length: tokens={total_tokens} batch={batch_size}"
                    )
                q_len = total_tokens // batch_size
                hidden_3d = hidden_states.reshape(
                    batch_size, q_len, hidden_states.shape[-1]
                )
                q_lora_3d = q_lora.reshape(batch_size, q_len, q_lora.shape[-1])
                positions = fmha_params.positions_d.reshape(-1).to(torch.long)
                topk_buffer = torch.full(
                    (batch_size, q_len, self.index_topk),
                    -1,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
                compressor_meta = None
                if q_len > 1:
                    from dataclasses import replace

                    b_idx = torch.arange(
                        batch_size, device=hidden_states.device, dtype=torch.long
                    ).repeat_interleave(q_len)
                    cu_seq_per_req = torch.arange(
                        0,
                        total_tokens + 1,
                        q_len,
                        device=hidden_states.device,
                        dtype=torch.int32,
                    )
                    self.compressed_indexer._propagate_pool_to_nested()
                    compressor_meta = (
                        self.compressed_indexer.compressor.prepare_metadata(
                            positions,
                            b_idx,
                            has_prefix=True,
                            is_batched=True,
                            seq_start_per_req=positions.view(batch_size, q_len)[:, 0]
                            .to(torch.int32)
                            .contiguous(),
                            cu_seq_per_req=cu_seq_per_req,
                        )
                    )
                    compressor_meta = replace(
                        compressor_meta,
                        compressed_lens_per_token=(
                            (positions + 1) // self.compress_ratio
                        ).to(torch.int32),
                    )
                topk = self.compressed_indexer.forward_decode_vectorized(
                    hidden_3d,
                    q_lora_3d,
                    positions.view(batch_size, q_len)[:, 0],
                    topk_buffer,
                    position_ids=positions if q_len > 1 else None,
                    compressor_meta=compressor_meta,
                ).reshape(-1, self.index_topk)
            return None if use_fast_path else topk
        finally:
            self.compressed_indexer.clear_pool_context()
            self.compressed_indexer.set_cp_ctx(None)
            self.compressed_indexer.compressor.set_cp_ctx(None)

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _prefill_cp_fused_quant_enabled(self) -> bool:
        return os.environ.get(
            "DSV4_CP_PREFILL_INDEXER_FUSED_QUANT", "1"
        ).strip().lower() in ("1", "true", "yes", "on")

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return bool(attention_inputs.is_prefill) and self._prefill_cp_enabled()

    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        # F3: fused (cast + GEMV + 2 elementwise muls) into one Triton kernel.
        # ``self._fuse_logits_head_gate`` is resolved at __init__ from
        # ``HWKernelConfig.enable_fuse_kernels``.
        scale = self.softmax_scale * self.weights_scale
        if self._fuse_logits_head_gate and x.is_contiguous():
            return fused_logits_head_gate(
                x,
                q_scale,
                self.weights_proj.weight,
                scale,
                fallback_proj=self.weights_proj,
            )
        x = x.float()
        weights = self.weights_proj(x)
        weights = weights.float()
        weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _fused_forward_decode(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused decode: QK kernel does K(RoPE+Had→bf16) + Q(RoPE+Had+FP8).

        Returns (q_fp8, q_scale).
        """
        q = _project_with_optional_fp8(self.wq_b, q_lora, q_c_fp8, q_c_scale)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = _project_with_optional_fp8(self.wk, x, x_fp8, x_scale)
        k = self.k_norm(k)

        q_fp8, q_scale, key = self.indexer_op.fused_rope_quant_qk(
            q, k, fmha_params.positions_d
        )
        self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)

        return q_fp8, q_scale

    def _fused_forward_prefill_cp(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        cp_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = _project_with_optional_fp8(self.wq_b, q_lora, q_c_fp8, q_c_scale)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = _project_with_optional_fp8(self.wk, x, x_fp8, x_scale)
        k = self.k_norm(k)

        q_fp8, q_scale, key = self.indexer_op.fused_rope_quant_qk(
            q,
            k,
            cp_params.full_rope_pos_ids,
        )
        slot_mapping = (
            cp_params.sharded_slot_mapping
            if bool(getattr(cp_params, "kv_cache_sharded", False))
            else fmha_params.slot_mapping
        )
        self.indexer_op.quant_k_cp_only(
            key,
            kv_cache,
            slot_mapping,
            cp_params.kv_restore_unpad_indices,
        )
        return q_fp8, q_scale

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
        cp_params: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = _project_with_optional_fp8(self.wq_b, q_lora, q_c_fp8, q_c_scale)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        k = _project_with_optional_fp8(self.wk, x, x_fp8, x_scale)
        k = self.k_norm(k)

        if self._prefill_cp_enabled():
            assert cp_params is not None
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                cp_params.full_rope_pos_ids,
            )
        else:
            positions = flashmla_params.positions_d
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None
            slot_mapping = (
                cp_params.sharded_slot_mapping
                if bool(getattr(cp_params, "kv_cache_sharded", False))
                else fmha_params.slot_mapping
            )
            return self.indexer_op.quant_q_k_cp(
                query,
                key,
                kv_cache,
                slot_mapping,
                cp_params.kv_restore_unpad_indices,
            )
        return self.indexer_op.quant_q_k(query, key, kv_cache, fmha_params.slot_mapping)

    def _compute_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> torch.Tensor:
        is_multi_token_decode = bool(
            getattr(attention_inputs, "is_target_verify", False)
        ) or bool(getattr(attention_inputs, "is_draft_extend", False))
        if not attention_inputs.is_prefill or is_multi_token_decode:
            return self.indexer_op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if self._prefill_cp_enabled():
            assert cp_params is not None
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.cu_kv_seqlens_global,
                cp_params.total_kv_len,
                cp_params.precomputed_ks,
                cp_params.precomputed_ke,
                cp_params.precomputed_lengths,
                cp_params.precomputed_topk_off,
                bool(getattr(cp_params, "kv_cache_sharded", False)),
                int(getattr(cp_params, "cp_size", 1)),
                int(getattr(cp_params, "cp_rank", 0)),
                kv_owner_tokens_per_block=self._kv_owner_tokens_per_block,
                indexer_cp_plan=getattr(cp_params, "indexer_cp_plan", None),
                indexer_cp_local_cu=getattr(cp_params, "indexer_cp_local_cu", None),
                indexer_copy_dst_idx=getattr(cp_params, "indexer_copy_dst_idx", None),
                indexer_src_for_padded=getattr(
                    cp_params, "indexer_src_for_padded", None
                ),
                total_local_ids_is_identity=bool(
                    getattr(cp_params, "total_local_ids_is_identity", False)
                ),
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: KVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
        global_kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        if self.compress_ratio > 1:
            if global_kv_cache is None:
                raise RuntimeError(
                    "GLM-5.3-Flash compressed indexer requires global KVCache"
                )
            return self._forward_compressed(
                hidden_states,
                q_lora,
                fmha_params,
                attention_inputs,
                global_kv_cache,
                use_fast_path,
                cp_params,
            )
        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params is required for sparse prefill CP"

        # Fused Q-RoPE-Hadamard-Quant path: single Triton kernel does
        # RoPE + 128-pt Hadamard + ue8m0 FP8 quant for Q (decode only).
        if (
            self._is_sparse_prefill_cp(attention_inputs)
            and self._prefill_cp_fused_quant_enabled()
            and cp_params.full_rope_pos_ids is not None
        ):
            q_fp8, q_scale = self._fused_forward_prefill_cp(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        elif (
            self._fuse_logits_head_gate
            and not attention_inputs.is_prefill
            and cp_params is None
        ):
            q_fp8, q_scale = self._fused_forward_decode(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        else:
            query, key = self._get_q_k_bf16(
                q_lora,
                hidden_states,
                fmha_params,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
            q_fp8, q_scale = self._quantize_q_k(
                query, key, kv_cache, fmha_params, attention_inputs, cp_params
            )

        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )


def bind_indexer_block_table_group_ids(
    layers: Any, kv_cache: Optional[KVCache]
) -> None:
    """Resolve physical block-table groups once and bind compressed indexers."""
    compressed_indexers = []
    for layer in layers:
        indexer = getattr(getattr(layer, "self_attn", None), "indexer", None)
        if isinstance(indexer, Indexer) and indexer.compress_ratio > 1:
            compressed_indexers.append(indexer)

    if not compressed_indexers or kv_cache is None:
        return

    from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
        resolve_block_table_group_ids,
    )

    table_group_ids = resolve_block_table_group_ids(kv_cache)
    for indexer in compressed_indexers:
        indexer._block_table_group_ids = table_group_ids
