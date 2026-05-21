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
from rtp_llm.ops import (
    AttentionConfigs,
    FMHAConfig,
    FMHAType,
    KvCacheDataType,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs, rtp_llm_ops
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

    def _convert_topk_indices_to_global(
        self, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """Request-local topk → physical positions in the flat paged cache.

        Returns [T, 1, topk]. h_kv=1 for MLA — heads share indices.
        """
        assert self.block_table is not None and self.mla_params is not None
        topk_2d = _topk_2d(topk_indices)
        topk = topk_2d.shape[1]
        assert topk == self.top_k, f"topk {topk} != top_k {self.top_k}"
        global_2d = triton_convert_req_index_to_global_index(
            req_id=self.mla_params.batch_indice_d,
            block_table=self.block_table,
            token_indices=topk_2d,
            BLOCK_SIZE=self.token_per_block,
            NUM_TOPK_TOKENS=topk,
            BLOCK_N=min(128, topk),
            HAS_PREFILL_WORKSPACE=False,
        )
        return global_2d.unsqueeze(1)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
        layer_id: int = 0,
    ) -> torch.Tensor:
        """q: [T, H, qk_head_dim], kv: [total_kv_len, 1, kv_lora_rank+rope].

        Returns [T, H, kv_lora_rank].
        """
        global_indices = self._convert_topk_indices_to_global(topk_indices)
        out, _, _ = flash_mla_sparse_fwd(
            q, kv, global_indices, self.scale, d_v=self.kv_lora_rank
        )
        return out


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
    """Transient buffers for the gather + sparse_fwd prefill path.

    Allocated per-forward in plan(); reused across all layers in that forward.
    """

    fused_kv: torch.Tensor  # [total_kv_len, kv_lora_rank + rope], bf16
    workspace_starts: torch.Tensor  # [batch_size], int32, indptr[:-1]
    seq_lens: torch.Tensor  # [batch_size], int32, indptr diff
    total_kv_len: int
    batch_size: int


class SparseMlaFp8Op(SparseMlaOp):
    """FP8 sparse MLA. See module docstring for path selection."""

    expects_paged_kv = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Fresh sched_meta per forward (set in plan())
        self._sched_meta = None
        # Gather workspace (None when path is disabled / no work to do)
        self._gather: Optional[_GatherWorkspace] = None

    def plan(
        self,
        mla_params: rtp_llm_ops.FlashInferMlaAttnParams,
        block_table: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs] = None,
    ) -> None:
        super().plan(mla_params, block_table, attn_inputs)

        # get_mla_metadata returns an empty FlashMLASchedMeta; the kernel fills
        # it on first call, then reuses it for the rest of the forward.
        self._sched_meta, _ = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=mla_params.batch_indice_h.shape[0] * self.num_heads,
            topk=self.top_k,
            num_heads_q=self.num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )

        # Gather path: only for prefill, gated by env var.
        gather_enabled = (
            os.environ.get("USE_GATHER_PATH", "0") == "1"
            and attn_inputs is not None
            and getattr(attn_inputs, "is_prefill", False)
        )
        self._gather = self._build_gather_workspace() if gather_enabled else None

    def _build_gather_workspace(self) -> Optional[_GatherWorkspace]:
        """Slice the prefill indptr and allocate the BF16 workspace.

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
            fused_kv=torch.empty(
                (total_kv_len, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.bfloat16,
                device=self.block_table.device,
            ),
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
    ) -> torch.Tensor:
        if self._gather is not None:
            return self._forward_gather(q, kv, topk_indices)
        return self._forward_with_kvcache(q, kv, topk_indices)

    def _forward_gather(
        self,
        q: torch.Tensor,
        kv_cache_fp8: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """gather + flash_mla_sparse_fwd (prefill fast path)."""
        ws = self._gather
        assert (
            ws is not None
            and self.mla_params is not None
            and self.block_table is not None
        )

        # Cache as uint8, drop head dim if present → [num_blocks, block_size, 656]
        src = _as_uint8(kv_cache_fp8)
        if src.ndim == 4:
            src = src.squeeze(2)

        # FP8 paged → BF16 contiguous workspace
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache_v2(
            src,
            ws.fused_kv,
            self.block_table.to(torch.int32),
            ws.seq_lens,
            ws.workspace_starts,
            ws.batch_size,
            ws.total_kv_len,
        )

        # Request-local topk → workspace offset (ws_starts[req] + local_pos)
        offsets = ws.workspace_starts[self.mla_params.batch_indice_d]
        global_indices = (_topk_2d(topk_indices) + offsets.unsqueeze(1)).unsqueeze(1)

        out, _, _ = flash_mla_sparse_fwd(
            q,
            ws.fused_kv.unsqueeze(1),  # [total_kv_len, 1, dim]
            global_indices,
            self.scale,
            d_v=self.kv_lora_rank,
        )
        return out

    def _forward_with_kvcache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """flash_mla_with_kvcache directly on FP8 paged cache."""
        assert self._sched_meta is not None
        # Cache layout: (num_blocks, block_size, num_heads_k=1, dim)
        kv_cache = _as_uint8(kv)
        if kv_cache.ndim == 3:
            kv_cache = kv_cache.unsqueeze(-2)

        # Indices: [T, 1, topk] → [1, T, topk] (kernel expects batched layout)
        global_indices = (
            self._convert_topk_indices_to_global(topk_indices).squeeze(1).unsqueeze(0)
        )

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
        )
        return attn_out.squeeze(0)


# ---------------------------------------------------------------------------
# MLA layer wrapper: RoPE + KV write + input/output BMM + the operator above
# ---------------------------------------------------------------------------


class SparseMlaImpl(MlaImplBase):
    """Wraps a SparseMlaOp / SparseMlaFp8Op with rope, KV write, and absorbed BMMs."""

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
        self.fmha_impl: SparseMlaOp = op_cls(
            attn_configs.head_num,
            attn_configs.kv_lora_rank,
            attn_configs.rope_head_dim,
            attn_configs.nope_head_dim,
            attn_configs.kernel_tokens_per_block,
            attn_configs.softmax_extra_scale,
            attn_configs.indexer_topk,
            parallelism_config=parallelism_config,
        )

        self.rope_impl = NewMlaRotaryEmbeddingOp(
            cos_sin_cache=cos_sin_cache,
            is_neox_style=self.attn_configs.rope_config.is_neox_style,
        )
        self.kv_cache_write_op = MlaKVCacheWriteOp(
            kv_cache_dtype=attn_configs.kv_cache_dtype,
        )
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
        self.fmha_impl.plan(
            self.fmha_params,
            attn_inputs.kv_cache_kernel_block_id_device,
            attn_inputs=attn_inputs,
        )

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        self.prepare(attn_inputs, forbid_realloc=True)

    # -- BMMs ----------------------------------------------------------------

    def _apply_input_bmm(self, q: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Project q_nope @ W_kc to kv_lora_rank, assemble [T, H, kv_lora_rank|rope].

        q_pe is a strided view from torch.split — calling .contiguous() here would
        re-introduce the very copy the strided triton kernel is replacing.
        """
        q_nope, q_pe = q.view(
            -1, self.num_heads, self.nope_head_dim + self.rope_head_dim
        ).split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        q_transformed = torch.empty(
            q_nope.shape[0],
            self.num_heads,
            self.kv_lora_rank + self.rope_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        strided_slice_copy_(q_transformed, q_pe, self.kv_lora_rank)

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

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sparse MLA forward. q: [T, H, qk_head_dim], topk: [T, (H,) topk] (req-local).
        Returns [T, H, nope_head_dim]."""
        assert topk_indices is not None and kv_cache is not None

        # 1. RoPE on q_pe and k_pe; write KV to cache + optional store
        q_pe = q[:, :, self.nope_head_dim :]
        self.rope_impl.forward(q_pe, k_pe, self.rope_params)
        self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, self.rope_params)
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # 2. Project q via W_kc into the absorbed kv_lora_rank space
        q_transformed = self._apply_input_bmm(q, layer_id)

        # 3. Sparse attention. FP8 op consumes paged shape; BF16 op wants flat.
        if self.fmha_impl.expects_paged_kv:
            kv_input = kv_cache.kv_cache_base
        else:
            kv_input = kv_cache.kv_cache_base.view(
                -1, 1, kv_cache.kv_cache_base.size(-1)
            )
        attn_output = self.fmha_impl.forward(
            q_transformed, kv_input, topk_indices, layer_id=layer_id
        )

        # 4. Project attention output via W_vc → final output
        return self._apply_output_bmm(attn_output, layer_id)
