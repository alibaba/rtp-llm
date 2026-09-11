"""
Sparse MLA implementation for prefill and decode.

Two operators:
- SparseMlaOp:    BF16 KV cache → flash_mla_sparse_fwd
- SparseMlaFp8Op: FP8 paged KV cache. Two paths controlled by USE_GATHER_PATH env:
    * USE_GATHER_PATH=1 (prefill): gather + upconvert FP8 → BF16 workspace,
      then flash_mla_sparse_fwd. ~1.7x faster than with_kvcache for large s_q.
    * Otherwise: flash_mla_with_kvcache directly on FP8 paged cache.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

# flash_mla requires CUDA >= 12.9. On unsupported envs the symbols stay
# undefined and any caller using sparse MLA will fail fast at use time.
try:
    cuda_ver = torch.version.cuda or ""
    _major, _minor = (int(x) for x in (cuda_ver.split(".") + ["0", "0"])[:2])
    if (_major, _minor) >= (12, 9):
        from flash_mla import (
            flash_mla_sparse_fwd,
            flash_mla_with_kvcache,
            get_mla_metadata,
        )
except (ImportError, AttributeError, ValueError) as _e:
    logging.warning(f"flash_mla not available: {_e}. Requires CUDA >= 12.9")

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_kv_cache_write_op import (
    MlaKVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import MlaImplBase
from rtp_llm.models_py.triton_kernels.common.strided_slice_copy import (
    strided_slice_copy_,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.block_index_to_global import (
    triton_convert_req_index_to_global_index,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla import (
    fused_qk_rope_cat_cache_mla,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.pad_query_heads import (
    maybe_pad_query_heads,
)
from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    rtp_llm_ops,
)
from rtp_llm.utils.model_weight import W

from .rope_emb_new import NewMlaRotaryEmbeddingOp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topk_2d(topk_indices: torch.Tensor) -> torch.Tensor:
    """[T, topk] or [T, h_kv, topk] → [T, topk]. MLA always has h_kv=1."""
    return topk_indices if topk_indices.dim() == 2 else topk_indices[:, 0, :]


def _as_uint8(kv: torch.Tensor) -> torch.Tensor:
    """Reinterpret an FP8 tensor as uint8 (no-op if already uint8)."""
    return kv.view(torch.uint8) if kv.dtype != torch.uint8 else kv


def _is_multi_token_decode(attn_inputs: PyAttentionInputs) -> bool:
    return bool(getattr(attn_inputs, "is_target_verify", False)) or bool(
        getattr(attn_inputs, "is_draft_extend", False)
    )


def _allocate_prefill_fused_kv(
    total_kv_len: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate the layer-local BF16 gather destination after Indexer."""
    return torch.empty((total_kv_len, width), dtype=torch.bfloat16, device=device)


def _fp8_sparse_padded_heads(num_heads: int) -> int:
    """Return the 64/128-head envelope accepted by FP8 sparse FlashMLA."""
    return 64 if num_heads <= 64 else 128


def _is_sm100_or_newer(device: Optional[torch.device] = None) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(device)[0] >= 10
    except (AssertionError, RuntimeError):
        return False


def _bf16_sparse_padded_heads(
    num_heads: int, device: Optional[torch.device] = None
) -> int:
    """Return the head envelope required by BF16 sparse FlashMLA prefill."""
    padding_multiple = 128 if _is_sm100_or_newer(device) else 64
    return (num_heads + padding_multiple - 1) // padding_multiple * padding_multiple


# ---------------------------------------------------------------------------
# BF16 sparse MLA operator
# ---------------------------------------------------------------------------


class SparseMlaOp(object):
    """BF16 sparse MLA: flash_mla_sparse_fwd on a flat KV buffer."""

    def __init__(
        self,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        qk_nope_head_dim: int,
        page_size: int,
        softmax_extra_scale: float,
        top_k: int,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.token_per_block = page_size
        self.softmax_extra_scale = softmax_extra_scale
        self.scale = (self.qk_head_dim**-0.5) * softmax_extra_scale
        self.top_k = top_k

        # Filled by plan() each forward
        self.block_table: Optional[torch.Tensor] = None
        self.mla_params: Optional[rtp_llm_ops.FlashInferMlaAttnParams] = None
        # Several HY4 layers consume the exact same request-local TopK tensor.
        # Its physical-cache conversion is also identical while the block table
        # and request-id mapping stay unchanged, so retain the most recent
        # result for the following shared-TopK layers. plan() clears this state
        # at every forward boundary.
        self._global_topk_source: Optional[torch.Tensor] = None
        self._global_topk_block_table: Optional[torch.Tensor] = None
        self._global_topk_req_ids: Optional[torch.Tensor] = None
        self._global_topk_result: Optional[torch.Tensor] = None

    # Sub-classes that consume KV in paged layout override this to True.
    expects_paged_kv: bool = False

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        self.block_table = block_table
        self.mla_params = mla_params
        self._global_topk_source = None
        self._global_topk_block_table = None
        self._global_topk_req_ids = None
        self._global_topk_result = None

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Request-local topk → physical positions in the flat paged cache.

        Returns [T, 1, topk]. h_kv=1 for MLA — heads share indices.
        """
        assert self.block_table is not None and self.mla_params is not None
        req_ids = self.mla_params.batch_indice_d
        if (
            topk_indices is self._global_topk_source
            and self.block_table is self._global_topk_block_table
            and req_ids is self._global_topk_req_ids
            and self._global_topk_result is not None
        ):
            return self._global_topk_result

        topk_2d = _topk_2d(topk_indices)
        topk = topk_2d.shape[1]
        assert topk == self.top_k, f"topk {topk} != top_k {self.top_k}"
        global_2d = triton_convert_req_index_to_global_index(
            req_id=req_ids,
            block_table=self.block_table,
            # REBASE CONFLICT CONTEXT(e2e00e570): source branch passed
            # `token_indices=topk_2d, BLOCK_SIZE=self.token_per_block`; new
            # base renamed the kernel parameters to separate block-table tokens
            # from physical cache entries. Keep the new API and the source
            # branch's 2D topk normalization.
            token_indices=topk_2d,
            TOKENS_PER_BLOCK_FOR_BLOCK_TABLE=self.token_per_block,
            ENTRIES_PER_BLOCK=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        result = global_2d.unsqueeze(1)
        self._global_topk_source = topk_indices
        self._global_topk_block_table = self.block_table
        self._global_topk_req_ids = req_ids
        self._global_topk_result = result
        return result

    def _pad_query_and_sink(
        self,
        q: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        kernel_heads: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Pad HY V4 TP-local heads to FlashMLA's 64/128-head ABI.

        Padded sinks are ``-inf`` so they add exactly zero to the softmax
        denominator. A zero sink would incorrectly create an extra exp(0).
        """
        actual_heads = q.size(1)
        kernel_heads = self.num_heads if kernel_heads is None else kernel_heads
        if actual_heads > kernel_heads:
            raise ValueError(
                f"query has {actual_heads} heads but FlashMLA was planned for "
                f"{kernel_heads}"
            )
        if attn_sink is not None and attn_sink.numel() != actual_heads:
            raise ValueError(
                f"attention sink has {attn_sink.numel()} heads, expected {actual_heads}"
            )
        if attn_sink is not None and attn_sink.device != q.device:
            raise ValueError(
                "attention sink and query must be on the same device, got "
                f"{attn_sink.device} and {q.device}"
            )
        if actual_heads == kernel_heads:
            return q, attn_sink, actual_heads
        q_padded = (
            maybe_pad_query_heads(q, kernel_heads) if fuse_kernels_enabled() else None
        )
        if q_padded is None:
            q_padded = q.new_zeros((q.size(0), kernel_heads, q.size(2)))
            q_padded[:, :actual_heads].copy_(q)
        if attn_sink is not None:
            sink_padded = torch.full(
                (kernel_heads,),
                -torch.inf,
                dtype=torch.float32,
                device=attn_sink.device,
            )
            sink_padded[:actual_heads].copy_(attn_sink)
            attn_sink = sink_padded
        return q_padded, attn_sink, actual_heads

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        physical_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """q: [T, H, qk_head_dim], kv: [total_kv_len, 1, kv_lora_rank+rope].

        Returns [T, H, kv_lora_rank].
        """
        q, attn_sink, actual_heads = self._pad_query_and_sink(q, attn_sink)
        global_indices = (
            self._convert_topk_indices_to_global(topk_indices)
            if physical_indices is None
            else physical_indices
        )
        sink_kwargs = {} if attn_sink is None else {"attn_sink": attn_sink}
        out, _, _ = flash_mla_sparse_fwd(
            q,
            kv,
            global_indices,
            self.scale,
            d_v=self.kv_lora_rank,
            **sink_kwargs,
        )
        return out[:, :actual_heads]


# ---------------------------------------------------------------------------
# FP8 sparse MLA operator (gather path + with_kvcache fallback)
# ---------------------------------------------------------------------------


class SparseMlaFp8DecodeParams(object):
    """Wraps the (sched_meta, num_splits) returned by get_mla_metadata.

    Kept as a plain class (not a dataclass) for backwards compatibility — the
    CP variant in flashmla_sparse_cp_impl.py imports this name.
    """

    def __init__(self, tile_scheduler_metadata, num_splits):
        self.tile_scheduler_metadata = tile_scheduler_metadata
        self.num_splits = num_splits


@dataclass
class _GatherWorkspace:
    """Metadata for the gather + sparse_fwd prefill path.

    plan() only records the layout. The large BF16 fused-KV tensor is allocated
    in forward(), after the indexer logits have gone out of scope, so the CUDA
    caching allocator can reuse that block instead of keeping both active.
    """

    workspace_starts: torch.Tensor  # [batch_size], int32, indptr[:-1]
    seq_lens: torch.Tensor  # [batch_size], int32, indptr diff
    total_kv_len: int
    batch_size: int


@dataclass
class _SparseMlaPreparedForward:
    """Top-K-independent SparseMLA work submitted before Indexer completion."""

    q_transformed: torch.Tensor
    kv_input: torch.Tensor
    layer_id: int
    attn_sink: Optional[torch.Tensor]
    physical_indices: Optional[torch.Tensor] = None
    pinned_cache: Optional[tuple] = None


class SparseMlaFp8Op(SparseMlaOp):
    """FP8 sparse MLA. See module docstring for path selection."""

    expects_paged_kv = True

    def __init__(self, *args, **kwargs):
        self.use_cuda_graph = bool(kwargs.pop("use_cuda_graph", False))
        bf16_prefill_num_heads = kwargs.pop("bf16_prefill_num_heads", None)
        super().__init__(*args, **kwargs)
        self.bf16_num_heads = (
            _bf16_sparse_padded_heads(self.num_heads)
            if bf16_prefill_num_heads is None
            else int(bf16_prefill_num_heads)
        )
        # In CUDA graph mode the captured kernels keep the scheduler storage
        # address. Replacing this object during replay leaves the graph with a
        # dangling/stale pointer, so plan() must refresh it in place.
        self._sched_meta = None
        self._sched_meta_key = None
        # Gather workspace (None when path is disabled / no work to do)
        self._gather: Optional[_GatherWorkspace] = None

    def _reset_sched_meta(self, num_q_tokens_per_head_k: int) -> None:
        key = (
            int(num_q_tokens_per_head_k),
            int(self.top_k),
            int(self.num_heads),
            1,
            True,
        )
        if self._sched_meta is None or (
            not self.use_cuda_graph and self._sched_meta_key != key
        ):
            self._sched_meta, _ = get_mla_metadata(
                cache_seqlens=None,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                topk=self.top_k,
                num_heads_q=self.num_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
            )
            self._sched_meta_key = key
        elif self.use_cuda_graph and self._sched_meta_key != key:
            raise ValueError(
                "Sparse MLA FP8 CUDA graph replay changed scheduler shape: "
                f"captured={self._sched_meta_key}, current={key}"
            )

        if (
            self.use_cuda_graph
            and self._sched_meta_key == key
            and getattr(self._sched_meta, "tile_scheduler_metadata", None) is not None
            and getattr(self._sched_meta, "num_splits", None) is not None
        ):
            return

        self._sched_meta.tile_scheduler_metadata = None
        self._sched_meta.num_splits = None

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        super().plan(mla_params, block_table, attn_inputs)

        # get_mla_metadata returns an empty FlashMLASchedMeta; the kernel fills
        # it on first call, then reuses it for the rest of the forward.
        self._reset_sched_meta(int(mla_params.batch_indice_h.shape[0]) * self.num_heads)

        tiered = int(os.environ.get("RTP_LLM_DSA_MLA_HOST_CACHE_MB", "0")) > 0
        # Full history is gathered directly from both tiers for eager prefill.
        gather_enabled = (
            (os.environ.get("USE_GATHER_PATH", "0") == "1" or tiered)
            and attn_inputs is not None
            and getattr(attn_inputs, "is_prefill", False)
            and not _is_multi_token_decode(attn_inputs)
            and (not self.use_cuda_graph or tiered)
        )
        self._gather = self._build_gather_workspace() if gather_enabled else None

    def _build_gather_workspace(self) -> Optional[_GatherWorkspace]:
        """Slice the prefill indptr and record the gather layout.

        prefill_ragged_kv_len_indptr_d = [0, kv_len_0, kv_len_0+kv_len_1, ...].
        The buffer can be longer than the actual batch, hence the [:batch+1] slice.
        Returns None if total_kv_len == 0 (no prefill tokens).
        """
        assert self.mla_params is not None and self.block_table is not None
        batch_size = int(self.block_table.shape[0])
        indptr = self.mla_params.prefill_ragged_kv_len_indptr_d[: batch_size + 1]
        total_kv_len = int(indptr[batch_size].item())
        if total_kv_len == 0:
            return None
        return _GatherWorkspace(
            workspace_starts=indptr[:batch_size],
            seq_lens=indptr[1:] - indptr[:batch_size],
            total_kv_len=total_kv_len,
            batch_size=batch_size,
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
        layer_id: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        physical_indices: Optional[torch.Tensor] = None,
        pinned_cache: Optional[tuple] = None,
    ) -> torch.Tensor:
        if self._gather is not None and physical_indices is None:
            return self._forward_gather(q, kv, topk_indices, attn_sink, pinned_cache)
        return self._forward_with_kvcache(
            q,
            kv,
            topk_indices,
            layer_id,
            attn_sink=attn_sink,
            physical_indices=physical_indices,
        )

    def _forward_gather(
        self,
        q: torch.Tensor,
        kv_cache_fp8: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_sink: Optional[torch.Tensor] = None,
        pinned_cache: Optional[tuple] = None,
    ) -> torch.Tensor:
        """gather + flash_mla_sparse_fwd (prefill fast path)."""
        q, attn_sink, actual_heads = self._pad_query_and_sink(
            q,
            attn_sink,
            self.bf16_num_heads,
        )
        ws = self._gather
        assert (
            ws is not None
            and self.mla_params is not None
            and self.block_table is not None
        )

        # Layer-local lifetime lets the caching allocator reuse indexer logits
        # storage. No synchronize/empty_cache is needed (or desirable).
        fused_kv = _allocate_prefill_fused_kv(
            ws.total_kv_len,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.block_table.device,
        )

        # Cache as uint8, drop head dim if present → [num_blocks, block_size, 656]
        src = _as_uint8(kv_cache_fp8)
        if src.ndim == 4:
            src = src.squeeze(2)

        # FP8 paged → BF16 contiguous workspace
        if pinned_cache is not None:
            working, group_layer = pinned_cache
            working.gather_bf16(
                group_layer,
                fused_kv,
                self.block_table,
                ws.seq_lens,
                ws.workspace_starts,
            )
        else:
            rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
                src,
                fused_kv,
                self.block_table.to(torch.int32),
                ws.seq_lens,
                ws.workspace_starts,
                ws.batch_size,
                ws.total_kv_len,
            )

        # Request-local topk → workspace offset (ws_starts[req] + local_pos)
        offsets = ws.workspace_starts[self.mla_params.batch_indice_d]
        topk_2d = _topk_2d(topk_indices)
        padding_mask = topk_2d < 0
        raw_global = topk_2d + offsets.unsqueeze(1)
        global_indices = raw_global.masked_fill(padding_mask, -1).unsqueeze(1)

        sink_kwargs = {} if attn_sink is None else {"attn_sink": attn_sink}
        out, _, _ = flash_mla_sparse_fwd(
            q,
            fused_kv.unsqueeze(1),  # [total_kv_len, 1, dim]
            global_indices,
            self.scale,
            d_v=self.kv_lora_rank,
            **sink_kwargs,
        )
        return out[:, :actual_heads]

    def _forward_with_kvcache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        layer_id: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        physical_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """flash_mla_with_kvcache directly on FP8 paged cache."""
        q, attn_sink, actual_heads = self._pad_query_and_sink(q, attn_sink)
        assert self._sched_meta is not None
        if layer_id == 0:
            self._sched_meta.tile_scheduler_metadata = None
            self._sched_meta.num_splits = None
        # Cache layout: (num_blocks, block_size, num_heads_k=1, dim)
        kv_cache = _as_uint8(kv)
        if kv_cache.ndim == 3:
            kv_cache = kv_cache.unsqueeze(-2)

        # Indices: [T, 1, topk] → [1, T, topk] (kernel expects batched layout)
        global_indices = (
            (
                self._convert_topk_indices_to_global(topk_indices)
                if physical_indices is None
                else physical_indices
            )
            .squeeze(1)
            .unsqueeze(0)
        )

        sink_kwargs = {} if attn_sink is None else {"attn_sink": attn_sink}
        attn_out, _ = flash_mla_with_kvcache(
            q=q.unsqueeze(0),
            k_cache=kv_cache,
            block_table=self.block_table,
            head_dim_v=self.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=self._sched_meta,
            num_splits=None,
            is_fp8_kvcache=True,
            indices=global_indices,
            softmax_scale=self.scale,
            **sink_kwargs,
        )
        return attn_out.squeeze(0)[:, :actual_heads]


# ---------------------------------------------------------------------------
# MLA layer wrapper: RoPE + KV write + input/output BMM + the operator above
# ---------------------------------------------------------------------------


class SparseMlaImpl(MlaImplBase):
    """Wraps a SparseMlaOp / SparseMlaFp8Op with rope, KV write, and absorbed BMMs."""

    supports_topk_late_binding = True

    def can_fuse_kv_norm_cache(
        self, compressed_kv: torch.Tensor, kv_norm_weight: torch.Tensor
    ) -> bool:
        """Whether the cache writer can consume raw KV-A and apply RMSNorm."""
        return bool(
            self._fuse_qk_rope_cat_cache_mla
            and self._kv_cache_type == "fp8_ds_mla"
            and self.kv_lora_rank == 512
            and compressed_kv.dim() == 2
            and compressed_kv.dtype == torch.bfloat16
            and compressed_kv.stride(-1) == 1
            and kv_norm_weight.shape == (self.kv_lora_rank,)
            and kv_norm_weight.dtype == torch.bfloat16
            and kv_norm_weight.is_contiguous()
            and compressed_kv.device == kv_norm_weight.device
        )

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
        fmha_impl: Optional[type] = None,
    ) -> None:
        super().__init__(
            attn_configs=attn_configs,
            attn_inputs=attn_inputs,
            weights=weights,
            cos_sin_cache=cos_sin_cache,
            fmha_config=fmha_config,
            use_trt_fmha=use_trt_fmha,
            quant_config=quant_config,
            max_seq_len=max_seq_len,
            is_cuda_graph=is_cuda_graph,
            parallelism_config=parallelism_config,
        )
        self.seq_size_per_block = attn_configs.kernel_tokens_per_block
        self.num_heads = attn_configs.head_num
        self.kv_lora_rank = attn_configs.kv_lora_rank
        self.rope_head_dim = attn_configs.rope_head_dim
        self.nope_head_dim = attn_configs.nope_head_dim
        self.is_prefill = attn_inputs.is_prefill
        self.parallelism_config = parallelism_config

        # Pick the right op class
        if fmha_impl is not None:
            op_cls = fmha_impl
        elif attn_configs.kv_cache_dtype == KvCacheDataType.BASE:
            op_cls = SparseMlaOp
        elif attn_configs.kv_cache_dtype == KvCacheDataType.FP8:
            op_cls = SparseMlaFp8Op
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {attn_configs.kv_cache_dtype}"
            )
        op_kwargs = {"parallelism_config": parallelism_config}
        if issubclass(op_cls, SparseMlaFp8Op):
            op_kwargs["use_cuda_graph"] = is_cuda_graph
        has_hy4_sink = any(W.hy4_attn_sink in layer for layer in weights)
        kernel_num_heads = attn_configs.head_num
        if has_hy4_sink:
            # HY4's SM100 FlashMLA wheel has native h64/d576/sink kernels.
            # Avoid the compatibility h128 lanes; the h64 specialization is
            # mathematically equivalent within BF16 rounding error.
            kernel_num_heads = _fp8_sparse_padded_heads(kernel_num_heads)
            if issubclass(op_cls, SparseMlaFp8Op):
                op_kwargs["bf16_prefill_num_heads"] = kernel_num_heads
        self.fmha_impl: SparseMlaOp = op_cls(
            kernel_num_heads,
            attn_configs.kv_lora_rank,
            attn_configs.rope_head_dim,
            attn_configs.nope_head_dim,
            attn_configs.kernel_tokens_per_block,
            attn_configs.softmax_extra_scale,
            attn_configs.indexer_topk,
            **op_kwargs,
        )

        self.rope_impl = NewMlaRotaryEmbeddingOp(
            cos_sin_cache=cos_sin_cache,
            is_neox_style=self.attn_configs.rope_config.is_neox_style,
        )
        self.kv_cache_write_op = MlaKVCacheWriteOp(
            kv_cache_dtype=attn_configs.kv_cache_dtype,
        )

        self._fuse_qk_rope_cat_cache_mla = fuse_kernels_enabled()
        self._kv_cache_type = (
            "fp8_ds_mla"
            if attn_configs.kv_cache_dtype == KvCacheDataType.FP8
            else "auto"
        )
        self._cos_sin_cache = cos_sin_cache
        self._is_neox_style = attn_configs.rope_config.is_neox_style
        self._cuda_dag_indexer_metadata = None

        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

        # create_params is a hook subclasses (e.g. SparseMlaCpImpl) override
        # to attach CP-specific state after the base prepare(). Keep the call
        # — do not inline.
        self.create_params(attn_inputs)

    def create_params(self, attn_inputs: PyAttentionInputs) -> None:
        """Allocate fmha_params and run the first prepare(). Override hook."""
        self.fmha_params = rtp_llm_ops.SparseMlaParams()
        self.rope_params = self.fmha_params
        self.prepare(attn_inputs)

    def _refresh_paged_mqa_schedule_metadata(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool
    ) -> None:
        """Refresh DeepGEMM paged-MQA schedule metadata for graph replay.

        The target-verify DSA indexer uses the paged logits kernel even though
        the outer attention input is prefill-shaped. CUDA graph replay must keep
        the captured schedule tensor address stable, so replay updates it in
        place instead of replacing the tensor object.
        """
        if not (
            _is_multi_token_decode(attn_inputs)
            or not bool(getattr(attn_inputs, "is_prefill", False))
        ):
            return
        try:
            import deep_gemm
        except Exception:
            return
        if not hasattr(deep_gemm, "get_paged_mqa_logits_metadata"):
            return

        if _is_multi_token_decode(attn_inputs):
            lengths = self.fmha_params.expanded_seq_lens
        else:
            lengths = self.fmha_params.kvlen_d
        if not isinstance(lengths, torch.Tensor) or lengths.numel() == 0:
            return
        lengths_2d = lengths.reshape(-1, 1)
        new_schedule = deep_gemm.get_paged_mqa_logits_metadata(
            lengths_2d, self.seq_size_per_block, deep_gemm.get_num_sms()
        )

        current = getattr(self.fmha_params, "schedule_metadata", None)
        has_current = False
        if isinstance(current, torch.Tensor):
            try:
                has_current = current.numel() > 0
            except RuntimeError:
                has_current = False
        if has_current:
            if tuple(current.shape) != tuple(new_schedule.shape):
                if forbid_realloc:
                    raise RuntimeError(
                        "Sparse MLA paged-MQA schedule metadata shape changed "
                        f"during CUDA graph replay: captured={tuple(current.shape)}, "
                        f"current={tuple(new_schedule.shape)}"
                    )
                self.fmha_params.schedule_metadata = new_schedule
            else:
                current.copy_(new_schedule)
        else:
            if forbid_realloc:
                raise RuntimeError(
                    "Sparse MLA paged-MQA schedule metadata was not captured before "
                    "CUDA graph replay"
                )
            self.fmha_params.schedule_metadata = new_schedule

    # -- Hooks expected by MlaImplBase --------------------------------------

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.SPARSE_FLASHMLA

    @staticmethod
    def is_sparse() -> bool:
        return True

    @classmethod
    def support(
        cls, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return (
            attn_configs.is_sparse
            and attn_configs.use_mla
            and attn_configs.kv_cache_dtype
            in (KvCacheDataType.BASE, KvCacheDataType.FP8)
        )

    def prepare(
        self, attn_inputs: PyAttentionInputs, forbid_realloc: bool = False
    ) -> None:
        """Refresh per-forward params + plan. forbid_realloc=True under cuda graph replay."""
        self.fmha_params.fill_params(
            attn_inputs, self.seq_size_per_block, forbid_realloc
        )
        self._refresh_paged_mqa_schedule_metadata(attn_inputs, forbid_realloc)
        block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
        if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
            block_table = attn_inputs.kv_cache_block_id_device
        self.fmha_impl.plan(
            self.fmha_params,
            block_table,
            attn_inputs=attn_inputs,
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        if (
            _is_multi_token_decode(attn_inputs)
            and isinstance(self.fmha_impl, SparseMlaFp8Op)
            and attn_inputs.kv_cache_kernel_block_id_device is not None
            and self.fmha_params.multi_token_decode_total_tokens > 0
        ):
            block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_block_id_device
            self.fmha_params.fill_multi_token_decode_cuda_graph_params(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                block_table,
                self.seq_size_per_block,
            )
            self._refresh_paged_mqa_schedule_metadata(attn_inputs, forbid_realloc=True)
            self.fmha_impl.plan(self.fmha_params, block_table, attn_inputs=attn_inputs)
        # Decode fast path (draft model): SparseMlaImpl handles decode for sparse
        # configs because MlaFlashInferDecodeImpl rejects when is_sparse=True.
        # fillParams' 3 toHostContiguousI32 D2H syncs go away by delegating to
        # the device-only fillDecodeCudaGraphParams kernel.
        elif (
            not getattr(attn_inputs, "is_prefill", True)
            and isinstance(self.fmha_impl, SparseMlaOp)
            and attn_inputs.sequence_lengths_plus_1_d is not None
            and attn_inputs.kv_cache_kernel_block_id_device is not None
        ):
            block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_block_id_device
            self.fmha_params.fill_sparse_mla_decode_cuda_graph_params(
                attn_inputs.sequence_lengths_plus_1_d,
                block_table,
                self.seq_size_per_block,
            )
            self._refresh_paged_mqa_schedule_metadata(attn_inputs, forbid_realloc=True)
            self.fmha_impl.plan(self.fmha_params, block_table, attn_inputs=attn_inputs)
        elif (
            getattr(attn_inputs, "is_prefill", False)
            and not _is_multi_token_decode(attn_inputs)
            and isinstance(self.fmha_impl, SparseMlaFp8Op)
            and attn_inputs.kv_cache_kernel_block_id_device is not None
            and self.fmha_params.prefill_tokens_per_batch > 0
        ):
            block_table = getattr(attn_inputs, "kv_cache_kernel_block_id_device", None)
            if not isinstance(block_table, torch.Tensor) or block_table.numel() == 0:
                block_table = attn_inputs.kv_cache_block_id_device
            self.fmha_params.fill_target_verify_cuda_graph_params(
                attn_inputs.input_lengths,
                attn_inputs.prefix_lengths,
                block_table,
                self.seq_size_per_block,
            )
            self._refresh_paged_mqa_schedule_metadata(attn_inputs, forbid_realloc=True)
            self.fmha_impl.plan(self.fmha_params, block_table, attn_inputs=attn_inputs)
        else:
            self.prepare(attn_inputs, forbid_realloc=True)

        if self._cuda_dag_indexer_metadata is not None:
            self._cuda_dag_indexer_metadata.prepare()

    # -- BMMs ----------------------------------------------------------------

    def _apply_input_bmm(
        self,
        q: torch.Tensor,
        layer_id: int,
        q_transformed: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project q_nope @ W_kc to kv_lora_rank, assemble [T, H, kv_lora_rank|rope].

        q_pe is a strided view from torch.split — calling .contiguous() here would
        re-introduce the very copy the strided triton kernel is replacing.
        """
        q_nope, q_pe = q.view(
            -1, self.num_heads, self.nope_head_dim + self.rope_head_dim
        ).split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        if out is not None:
            if q_transformed is not None:
                raise ValueError(
                    "out and prefilled q_transformed are mutually exclusive"
                )
            expected_shape = (
                q_nope.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.rope_head_dim,
            )
            if (
                tuple(out.shape) != expected_shape
                or out.dtype != q.dtype
                or out.device != q.device
                or not out.is_contiguous()
            ):
                raise ValueError("invalid absorbed-query output buffer")
            q_transformed = out
            strided_slice_copy_(q_transformed, q_pe, self.kv_lora_rank)

        if q_transformed is None:
            q_transformed = torch.empty(
                q_nope.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.rope_head_dim,
                dtype=q.dtype,
                device=q.device,
            )
            strided_slice_copy_(q_transformed, q_pe, self.kv_lora_rank)
        else:
            expected_shape = (
                q_nope.shape[0],
                self.num_heads,
                self.kv_lora_rank + self.rope_head_dim,
            )
            if (
                q_transformed.shape != expected_shape
                or q_transformed.dtype != q.dtype
                or q_transformed.device != q.device
                or not q_transformed.is_contiguous()
            ):
                raise ValueError("invalid prefilled absorbed-query buffer")

        if q_nope.shape[0] > 0:
            k_weight = self.weights[layer_id][W.mla_kc]
            out_nope = q_transformed[..., : self.kv_lora_rank].transpose(0, 1)
            torch.bmm(q_nope.transpose(0, 1), k_weight, out=out_nope)  # type: ignore
        return q_transformed

    def _apply_output_bmm(
        self, attn_output: torch.Tensor, layer_id: int
    ) -> torch.Tensor:
        """Project [T, H, kv_lora_rank] @ W_vc → [T, H, v_head_dim].

        Allocates contiguous [T, H, V] and asks cuBLAS to write a transposed
        [H, T, V] view via strideC, eliminating a post-bmm .contiguous() copy.
        """
        v_weight = self.weights[layer_id][W.mla_vc]
        output = torch.empty(
            attn_output.shape[0],
            self.num_heads,
            v_weight.shape[-1],
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(attn_output.transpose(0, 1), v_weight, out=output.transpose(0, 1))
        return output

    # -- Main forward --------------------------------------------------------

    def uses_pinned_prefill_gather(self) -> bool:
        return (
            isinstance(self.fmha_impl, SparseMlaFp8Op)
            and self.fmha_impl._gather is not None
        )

    def prefetch_kv(self, layer_id: int, topk_indices: torch.Tensor) -> None:
        working, group_layer = self.pinned_mla_groups[layer_id]
        if group_layer == 0:
            working.begin(self.fmha_impl._convert_topk_indices_to_global(topk_indices))

    def prepare_hy4_native_query(
        self, q_nope, q_transformed, kv_cache, layer_id, attn_sink
    ) -> _SparseMlaPreparedForward:
        """Consume HY4 Q-B outputs using the shared BF16 absorbed-Q kernel."""
        from rtp_kernel.glm5 import absorbed_q_nope_bmm

        absorbed_q_nope_bmm(
            q_nope,
            self.weights[layer_id][W.mla_kc],
            out=q_transformed,
        )
        if layer_id not in self.pinned_mla_groups:
            common.apply_write_cache_store(
                self.write_cache_store_impl, self.attn_inputs, kv_cache
            )
        return _SparseMlaPreparedForward(
            q_transformed=q_transformed,
            kv_input=kv_cache.kv_cache_base,
            layer_id=layer_id,
            attn_sink=attn_sink,
        )

    def finish_hy4_cache_write(self, prepared, kv_cache, packed_rows):
        working, group_layer = self.pinned_mla_groups[prepared.layer_id]
        rows = self.rope_params.slot_mapping.numel()
        working.write(
            group_layer,
            self.rope_params.slot_mapping,
            packed_rows.view(-1, 656)[:rows].view(working.backing[group_layer].dtype),
        )
        prepared.kv_input = working.resident[group_layer]
        prepared.physical_indices = working.physical_indices
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

    def prepare_topk_independent_forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        attn_sink: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 0.0,
        q_transformed: Optional[torch.Tensor] = None,
    ) -> _SparseMlaPreparedForward:
        """Submit RoPE/cache write and absorbed-Q BMM before Top-K is ready."""
        assert kv_cache is not None
        working_entry = self.pinned_mla_groups.get(layer_id)
        cache_target = kv_cache
        write_slots = self.rope_params.slot_mapping
        if working_entry is not None:
            working, group_layer = working_entry
            cache_target = LayerKVCache()
            cache_target.kv_cache_base = torch.empty(
                (q.shape[0], 1, kv_cache.kv_cache_base.shape[-1]),
                dtype=kv_cache.kv_cache_base.dtype,
                device=q.device,
            )
            write_slots = torch.arange(q.shape[0], dtype=torch.int64, device=q.device)

        # 1. RoPE on q_pe and k_pe; write KV to cache + optional store
        q_pe = q[:, :, self.nope_head_dim :]
        if q_transformed is not None and (
            q_transformed.shape
            != (q.shape[0], self.num_heads, self.kv_lora_rank + self.rope_head_dim)
            or q_transformed.dtype != q.dtype
            or q_transformed.device != q.device
            or not q_transformed.is_contiguous()
        ):
            raise ValueError("invalid absorbed-query output buffer")
        if self._fuse_qk_rope_cat_cache_mla and kv_cache is not None:
            if q_transformed is None:
                q_transformed = torch.empty(
                    q.shape[0],
                    self.num_heads,
                    self.kv_lora_rank + self.rope_head_dim,
                    dtype=q.dtype,
                    device=q.device,
                )
            fused_qk_rope_cat_cache_mla(
                q=q,
                compressed_kv=compressed_kv,
                k_pe=k_pe,
                kv_cache=cache_target.kv_cache_base,
                slot_mapping=write_slots,
                positions=self.rope_params.positions_d,
                cos_sin_cache=self._cos_sin_cache,
                kv_lora_rank=self.kv_lora_rank,
                rope_head_dim=self.rope_head_dim,
                is_neox_style=self._is_neox_style,
                kv_cache_type=self._kv_cache_type,
                q_rope_output=q_transformed[..., self.kv_lora_rank :],
                kv_norm_weight=kv_norm_weight,
                kv_norm_eps=kv_norm_eps,
            )
        else:
            self.rope_impl.forward(q_pe, k_pe, self.rope_params)
            if q_transformed is not None:
                strided_slice_copy_(q_transformed, q_pe, self.kv_lora_rank)
            self.kv_cache_write_op.forward(
                compressed_kv,
                k_pe,
                cache_target,
                self.rope_params,
                slot_mapping_override=write_slots,
            )

        # 2. Project q via W_kc into the absorbed kv_lora_rank space
        q_transformed = self._apply_input_bmm(q, layer_id, q_transformed=q_transformed)

        physical_indices = None
        if working_entry is not None:
            working.write(
                group_layer,
                self.rope_params.slot_mapping,
                cache_target.kv_cache_base.flatten(0, 1),
            )
            kv_input = working.resident[group_layer]
            if not self.uses_pinned_prefill_gather():
                physical_indices = working.physical_indices
        else:
            kv_input = kv_cache.kv_cache_base
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )
        if not self.fmha_impl.expects_paged_kv:
            kv_input = kv_input.view(-1, 1, kv_input.size(-1))
        return _SparseMlaPreparedForward(
            q_transformed=q_transformed,
            kv_input=kv_input,
            layer_id=layer_id,
            attn_sink=attn_sink,
            physical_indices=physical_indices,
            pinned_cache=working_entry if self.uses_pinned_prefill_gather() else None,
        )

    def hy4_output_weight(self, layer_id: int) -> torch.Tensor:
        return self.weights[layer_id][W.mla_vc]

    def finish_hy4_native_attention(
        self, prepared: _SparseMlaPreparedForward, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Leave the output BMM to HY4's fused BMM/gate/quant epilogue."""
        return self.fmha_impl.forward(
            prepared.q_transformed,
            prepared.kv_input,
            topk_indices,
            layer_id=prepared.layer_id,
            attn_sink=prepared.attn_sink,
            **(
                {"physical_indices": prepared.physical_indices}
                if prepared.physical_indices is not None
                else {}
            ),
        )

    def finish_topk_dependent_forward(
        self,
        prepared: _SparseMlaPreparedForward,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run sparse attention and the output BMM after Top-K is available."""
        attn_output = self.fmha_impl.forward(
            prepared.q_transformed,
            prepared.kv_input,
            topk_indices,
            layer_id=prepared.layer_id,
            attn_sink=prepared.attn_sink,
            **(
                {"pinned_cache": prepared.pinned_cache}
                if prepared.pinned_cache is not None
                else {}
            ),
            **(
                {"physical_indices": prepared.physical_indices}
                if prepared.physical_indices is not None
                else {}
            ),
        )

        return self._apply_output_bmm(attn_output, prepared.layer_id)

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
        attn_sink: Optional[torch.Tensor] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 0.0,
        kv_prefetched: bool = False,
    ) -> torch.Tensor:
        """Sparse MLA forward. q: [T, H, qk_head_dim], topk: [T, (H,) topk] (req-local).
        Returns [T, H, nope_head_dim]."""
        assert topk_indices is not None
        if (
            layer_id in self.pinned_mla_groups
            and not kv_prefetched
            and not self.uses_pinned_prefill_gather()
        ):
            self.prefetch_kv(layer_id, topk_indices)
        prepared = self.prepare_topk_independent_forward(
            q,
            compressed_kv,
            k_pe,
            kv_cache,
            layer_id,
            attn_sink,
            kv_norm_weight,
            kv_norm_eps,
        )

        return self.finish_topk_dependent_forward(prepared, topk_indices)
