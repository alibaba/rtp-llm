"""MiniMax-M3.1 decode and target verification over packed NVFP4 cache.

This module deliberately adapts the SGLang demo's *numeric data flow* rather
than its slot-major pointer arithmetic:

* index-Q is cast to scale-1 E4M3 (Q8); the native Cake main-attention
  specialization consumes original BF16 Q, while legacy readers use Q8;
* persistent main K/V and index-K remain packed E2M1 with E4M3 block-16
  scales (KV4);
* index scoring and sparse main attention consume RTP's two opaque cache
  regions directly, without materializing BF16 history pages.

RTP stores one physical page per row and places K, V, their scales, packed
index-K, and index scales at explicit byte offsets.  SGLang stores independent
slot-major tensors.  Callers must therefore pass ``NVFP4CacheLayout`` instead
of reusing SGLang strides.

Ordinary decode and target verification share this path.  Score/top-k/partial/
output buffers are cached by physical bucket shape so the complete chain has
stable addresses under CUDA Graph.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import NVFP4CacheLayout
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_attention import (
    _q8kv4_sparse_decode_attention_mxfp8,
    q8kv4_sparse_decode_attention,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.nvfp4_q8_index_score import (
    q8kv4_index_score,
)


@dataclass(frozen=True)
class Q8KV4DecodeResult:
    output: torch.Tensor
    topk_indices: torch.Tensor
    index_scores: torch.Tensor


@dataclass(frozen=True)
class Q8KV4Mxfp8DecodeResult:
    output_fp8: torch.Tensor
    output_scales_packed: torch.Tensor
    topk_indices: torch.Tensor
    index_scores: torch.Tensor


@dataclass
class _Q8KV4DecodeWorkspace:
    q8: torch.Tensor
    idx_q8: torch.Tensor
    scores: torch.Tensor
    topk_i32: torch.Tensor
    output: torch.Tensor | None
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    partial_counts: torch.Tensor
    output_fp8: torch.Tensor | None
    output_scales_packed: torch.Tensor | None

    _CACHE: ClassVar[dict[tuple, "_Q8KV4DecodeWorkspace"]] = {}

    @classmethod
    def acquire(
        cls,
        q: torch.Tensor,
        idx_q: torch.Tensor,
        max_blocks: int,
        topk: int,
        num_topk_chunks: int,
        output_format: str = "bf16",
    ) -> "_Q8KV4DecodeWorkspace":
        batch, q_heads, dim = map(int, q.shape)
        idx_heads = int(idx_q.shape[1])
        key = (
            str(q.device),
            batch,
            q_heads,
            idx_heads,
            dim,
            max_blocks,
            topk,
            num_topk_chunks,
        )
        if output_format not in ("bf16", "mxfp8"):
            raise ValueError("unknown Q8KV4 output format")
        workspace = cls._CACHE.get(key)
        if workspace is None:
            workspace = cls(
                q8=torch.empty_like(q, dtype=torch.float8_e4m3fn),
                idx_q8=torch.empty_like(idx_q, dtype=torch.float8_e4m3fn),
                scores=torch.empty(
                    idx_heads,
                    batch,
                    max_blocks,
                    dtype=torch.float32,
                    device=q.device,
                ),
                topk_i32=torch.empty(
                    idx_heads,
                    batch,
                    topk,
                    dtype=torch.int32,
                    device=q.device,
                ),
                output=None,
                output_fp8=None,
                output_scales_packed=None,
                partial_output=torch.empty(
                    batch,
                    idx_heads,
                    topk,
                    q_heads // idx_heads,
                    dim,
                    dtype=torch.bfloat16,
                    device=q.device,
                ),
                partial_lse=torch.empty(
                    batch,
                    idx_heads,
                    topk,
                    q_heads // idx_heads,
                    dtype=torch.float32,
                    device=q.device,
                ),
                partial_counts=torch.empty(
                    batch,
                    idx_heads,
                    dtype=torch.int32,
                    device=q.device,
                ),
            )
            cls._CACHE[key] = workspace
        # Geometry buffers are shared by both output formats. Never replace an
        # allocated output: each captured format retains its own stable address.
        if output_format == "bf16":
            if workspace.output is None:
                workspace.output = torch.empty(
                    batch, q_heads, dim, dtype=torch.bfloat16, device=q.device
                )
        elif workspace.output_fp8 is None:
            aligned_m = (batch + 3) // 4 * 4
            output_fp8 = torch.empty(
                batch, q_heads, dim, dtype=torch.float8_e4m3fn, device=q.device
            )
            output_scales = torch.empty(
                (q_heads, aligned_m), dtype=torch.int32, device=q.device
            ).t()[:batch]
            # Publish together only after both allocations succeed.
            workspace.output_fp8 = output_fp8
            workspace.output_scales_packed = output_scales
        return workspace


def _scale1_e4m3(values: torch.Tensor, name: str, out: torch.Tensor) -> torch.Tensor:
    if values.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"{name} must be a floating query tensor, got {values.dtype}")
    if not values.is_cuda or values.ndim != 3 or not values.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous CUDA [batch,heads,dim], got "
            f"shape={tuple(values.shape)} strides={tuple(values.stride())}"
        )
    # M3.1 has no per-tensor or per-row Q scale.  PyTorch's E4M3FN cast is the
    # scale-1 Q8 contract used by the reference demo.
    out.copy_(values)
    return out


# Scratch belongs to a device stream, not a layer or a capture bucket. Layers
# and buckets execute serially on that stream. Separate streams cannot share
# mutable scale staging and addressing buffers.
_CAKE_SCALES: dict[tuple, object] = {}
_CAKE_WORKSPACES: dict[tuple, object] = {}


def _cake_supported(q, layout, topk, mma_scale_layout, max_blocks):
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        # W5 B1/B2 can regress on short histories even with a long reserved
        # table; do not read lengths back from GPU to select a graph backend.
        # Retain the old reader below the measured B4/W5 (20-row) boundary.
        and q.shape[0] >= 20
        and max_blocks > 4
        and tuple(q.shape[1:]) == (64, 128)
        and layout.num_heads == 4
        and layout.page_size == 128
        and layout.head_dim == 128
        and topk == 16
        and mma_scale_layout
        and torch.cuda.get_device_capability(q.device) == (10, 3)
    )


def _cake_attention(q, layout, block_table, topk_indices, seq_lens, **options):
    from .cake_attention import (
        CakeAttentionWorkspace,
        CakeScaleWorkspace,
        cake_paged_sparse_decode,
    )

    pages = int(layout.packed_main.shape[0])
    owner = (str(q.device), torch.cuda.current_stream(q.device).cuda_stream, pages)
    scales = _CAKE_SCALES.get(owner)
    if scales is None:
        scales = CakeScaleWorkspace.create(q.device, pages)
        _CAKE_SCALES[owner] = scales
    key = owner + (tuple(q.shape), tuple(block_table.shape), tuple(topk_indices.shape))
    workspace = _CAKE_WORKSPACES.get(key)
    if workspace is None:
        workspace = CakeAttentionWorkspace.create(
            q, layout, block_table, topk_indices, seq_lens, scale_workspace=scales
        )
        _CAKE_WORKSPACES[key] = workspace
    return cake_paged_sparse_decode(
        q, layout, block_table, topk_indices, seq_lens, workspace=workspace, **options
    )


def _logical_decode_block_table(
    block_table: torch.Tensor, block_size: int, max_seq_len: int | None
) -> torch.Tensor:
    # Graph metadata reserves extra speculative columns beyond the model's
    # logical context limit. Native TopK supports 8192 live pages, not those
    # unused capacity columns. Keep the strided view: no copy or device sync.
    if block_table.shape[1] <= 8192 or max_seq_len is None:
        return block_table
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    logical_blocks = (max_seq_len + block_size - 1) // block_size
    if logical_blocks > 8192:
        raise ValueError("Q8KV4 decode supports at most 8192 logical pages")
    return block_table[:, :logical_blocks]


@torch.no_grad()
def q8kv4_paged_sparse_decode(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    layout: NVFP4CacheLayout,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    indexer_dim: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
    sm_scale: float | None = None,
    idx_sm_scale: float | None = None,
    mma_scale_layout: bool = False,
    query_width: int = 1,
    query_cu_seqlens: torch.Tensor | None = None,
    max_query_width: int = 1,
    fuse_bf16_query_rounding: bool = False,
    bf16_main_query: bool = False,
    valid_token_mask: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> Q8KV4DecodeResult:
    """Run native Q8KV4; BF16 activation output remains the default API."""
    return _q8kv4_paged_sparse_decode(
        q,
        idx_q,
        layout,
        block_table,
        seq_lens,
        indexer_dim=indexer_dim,
        block_size=block_size,
        topk=topk,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        score_type=score_type,
        sm_scale=sm_scale,
        idx_sm_scale=idx_sm_scale,
        mma_scale_layout=mma_scale_layout,
        query_width=query_width,
        query_cu_seqlens=query_cu_seqlens,
        max_query_width=max_query_width,
        fuse_bf16_query_rounding=fuse_bf16_query_rounding,
        bf16_main_query=bf16_main_query,
        valid_token_mask=valid_token_mask,
        max_seq_len=max_seq_len,
    )


@torch.no_grad()
def _q8kv4_paged_sparse_decode_mxfp8(*args, **kwargs) -> Q8KV4Mxfp8DecodeResult:
    """Internal producer for the exact MXFP8 O-proj consumer; no public flag."""
    return _q8kv4_paged_sparse_decode(*args, **kwargs, _output_format="mxfp8")


@torch.no_grad()
def _q8kv4_paged_sparse_decode(
    q: torch.Tensor,
    idx_q: torch.Tensor,
    layout: NVFP4CacheLayout,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    indexer_dim: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
    sm_scale: float | None = None,
    idx_sm_scale: float | None = None,
    mma_scale_layout: bool = False,
    query_width: int = 1,
    query_cu_seqlens: torch.Tensor | None = None,
    max_query_width: int = 1,
    fuse_bf16_query_rounding: bool = False,
    bf16_main_query: bool = False,
    valid_token_mask: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    _output_format: str = "bf16",
) -> Q8KV4DecodeResult | Q8KV4Mxfp8DecodeResult:
    """Run RTP Q8KV4 decode or grouped target verification.

    M3.1 has one local index-Q head per local KV head.  Other index-head
    reduction contracts require a separate native-kernel mapping. Grouped
    verification keeps token-row block tables and causal sequence lengths;
    all query rows in one request must share the same physical page table.

    fuse_bf16_query_rounding replaces saturating BF16 pre-rounding plus the
    scale-1 copy, preserving the unrounded input carriers. Only MSA callers
    that previously pre-rounded both queries should opt in. The default keeps
    PyTorch conversion semantics for direct BF16/FP16/FP32 callers.

    max_seq_len is the enforced model context limit, not the Graph allocation
    capacity. Callers supplying it must guarantee every live seq_len is within
    that limit. It removes only speculative reserve columns above 8192 pages.
    """
    if fuse_bf16_query_rounding and (
        q.dtype != torch.bfloat16 or idx_q.dtype != torch.bfloat16
    ):
        raise ValueError("fused query rounding requires BF16 Q and index-Q")
    if score_type != "max":
        raise ValueError("Q8KV4 decode currently supports max index score only")
    if block_size != layout.page_size or block_size != 128:
        raise ValueError("Q8KV4 decode currently requires page=block=128")
    if q.shape[0] != idx_q.shape[0] or q.shape[-1] != layout.head_dim:
        raise ValueError("Q and index-Q batch/head dimensions do not match the cache")
    if idx_q.shape[1] != layout.num_heads:
        raise ValueError(
            "Q8KV4 decode requires one index-Q head per local KV head: "
            f"idx_heads={idx_q.shape[1]} kv_heads={layout.num_heads}"
        )
    if idx_q.shape[-1] != indexer_dim or indexer_dim != layout.head_dim:
        raise ValueError("Q8KV4 decode requires index_dim == main head_dim")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise ValueError("block_table must be int32 [batch,max_blocks]")
    if seq_lens.dtype != torch.int32 or seq_lens.shape != (q.shape[0],):
        raise ValueError("seq_lens must be int32 [batch]")
    if query_width < 1 or q.shape[0] % query_width:
        raise ValueError("query_width must be positive and divide the token batch")
    block_table = _logical_decode_block_table(block_table, block_size, max_seq_len)
    max_blocks = int(block_table.shape[1])
    target_chunks = max(
        1,
        min(topk, 1024 // max(1, int(q.shape[0]) * layout.num_heads)),
    )
    num_topk_chunks = 1 << (target_chunks.bit_length() - 1)
    if _output_format == "mxfp8" and (
        not q.is_cuda
        or torch.cuda.get_device_capability(q.device) != (10, 3)
        or not 0 < q.shape[0] <= 100
        or tuple(q.shape[1:]) != (64, 128)
        or layout.num_heads != 4
        or topk != 16
    ):
        raise ValueError(
            "MXFP8 output requires tested SM103 Hq64/Hkv4/D128/TopK16 M1..100"
        )
    workspace = _Q8KV4DecodeWorkspace.acquire(
        q, idx_q, max_blocks, topk, num_topk_chunks, _output_format
    )
    use_cake = bf16_main_query and _cake_supported(
        q, layout, topk, mma_scale_layout, max_blocks
    )
    if fuse_bf16_query_rounding:
        from .nvfp4_q8_query_cast import fused_query_cast

        fused_query_cast(
            q, idx_q, workspace.q8, workspace.idx_q8, cast_main_query=not use_cake
        )
        q8, idx_q8 = workspace.q8, workspace.idx_q8
    else:
        q8 = workspace.q8 if use_cake else _scale1_e4m3(q, "q", workspace.q8)
        idx_q8 = _scale1_e4m3(idx_q, "idx_q", workspace.idx_q8)
    logical = layout.logical_views(indexer_dim)
    score_fn = q8kv4_index_score
    score_options = {}
    if idx_q.shape[1] == 4:
        if query_cu_seqlens is not None:
            from .nvfp4_q8_grouped_index_score import q8kv4_ragged_grouped_index_score

            score_fn = q8kv4_ragged_grouped_index_score
            score_options.update(
                cu_seqlens=query_cu_seqlens,
                max_query_width=max_query_width,
            )
        elif 2 <= query_width <= 16:
            from .nvfp4_q8_grouped_index_score import q8kv4_grouped_index_score

            score_fn = q8kv4_grouped_index_score
            score_options["query_width"] = query_width
    index_scores = score_fn(
        idx_q8,
        logical.idx_k_fp4,
        logical.idx_k_scale,
        block_table,
        seq_lens,
        workspace.scores,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=indexer_dim**-0.5 if idx_sm_scale is None else idx_sm_scale,
        mma_scale_layout=mma_scale_layout,
        **score_options,
    )
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    # Reuse MiniMax-M3's production decode TopK.  IndexScore already encodes
    # mandatory init/local pages with the same 1e30/1e29 sentinels, and this
    # op writes int32 page ids directly into a graph-stable workspace.
    rtp_llm_ops.minimax_decode_topk(
        index_scores,
        seq_lens,
        workspace.topk_i32,
        int(block_size),
        int(topk),
    )
    topk_indices = workspace.topk_i32
    if use_cake:
        # Index-Q/idxK4 scoring and production TopK are unchanged. Cake reads
        # original BF16 main Q and packed cache. Preserve the existing packed
        # MXFP8 O-projection interface with the same RTP quantization epilogue.
        output = _cake_attention(
            q,
            layout,
            block_table,
            topk_indices,
            seq_lens,
            sm_scale=layout.head_dim**-0.5 if sm_scale is None else sm_scale,
            mma_scale_layout=mma_scale_layout,
            valid_token_mask=valid_token_mask,
        )
        if _output_format == "mxfp8":
            from .nvfp4_q8_combine_mxfp8 import _quantize_bf16_attention_mxfp8

            _quantize_bf16_attention_mxfp8[(q.shape[0], 4)](
                output,
                workspace.output_fp8.view(torch.uint8),
                workspace.output_scales_packed,
                ALIGNED_M=workspace.output_scales_packed.stride(1),
                num_warps=4,
            )
            return Q8KV4Mxfp8DecodeResult(
                workspace.output_fp8, workspace.output_scales_packed,
                topk_indices, index_scores,
            )
        return Q8KV4DecodeResult(output, topk_indices, index_scores)
    attention_fn = q8kv4_sparse_decode_attention
    attention_out = workspace.output
    output_options = {}
    if _output_format == "mxfp8":
        attention_fn = _q8kv4_sparse_decode_attention_mxfp8
        attention_out = workspace.output_fp8
        output_options["output_scales"] = workspace.output_scales_packed
    output = attention_fn(
        q8,
        logical.main_k_fp4,
        logical.main_v_fp4,
        logical.main_k_scale,
        logical.main_v_scale,
        block_table,
        topk_indices,
        seq_lens,
        sm_scale=layout.head_dim**-0.5 if sm_scale is None else sm_scale,
        out=attention_out,
        partial_out=workspace.partial_output,
        partial_lse=workspace.partial_lse,
        counts=workspace.partial_counts,
        mma_scale_layout=mma_scale_layout,
        valid_token_mask=valid_token_mask,
        **output_options,
    )
    if _output_format == "mxfp8":
        return Q8KV4Mxfp8DecodeResult(
            output_fp8=output,
            output_scales_packed=workspace.output_scales_packed,
            topk_indices=topk_indices,
            index_scores=index_scores,
        )
    return Q8KV4DecodeResult(
        output=output,
        topk_indices=topk_indices,
        index_scores=index_scores,
    )
