"""Batch-invariant BF16 projections, paged attention and expert execution."""

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from rtp_llm.config.cuda_graph import (
    CudaGraphSelectionMode,
    GenerationPrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.common import (
    copy_kv_cache_offset,
    reshape_paged_kv_cache,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops import AttentionConfigs, KvCacheDataType
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeKVCacheDecodeOp,
    FusedRopeKVCachePrefillOpQOut,
)

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP


@triton.jit(do_not_specialize=["M"])
def _linear(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    # Each output element is owned by one CTA. M only changes grid size and masks.
    for start in range(0, K, BK):
        inner = start + ks
        a = tl.load(
            x + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + inner[:, None] * STRIDE_WK + cols[None, :] * STRIDE_WN,
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        output + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


# Adapted for Qwen3 MLP projections from vLLM v0.33.0's persistent GEMM.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
@triton.jit(do_not_specialize=["M"])
def _persistent_mlp_linear(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SMS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    LARGE: tl.constexpr,
):
    # Qwen3's TP-local gate/up and down widths are multiples of 128.
    bm: tl.constexpr = 128
    bn: tl.constexpr = 128
    bk: tl.constexpr = 64
    group_m: tl.constexpr = 8
    row_tiles = tl.cdiv(M, bm)
    col_tiles: tl.constexpr = N // bn
    group_tiles: tl.constexpr = group_m * col_tiles
    for tile in tl.range(tl.program_id(0), row_tiles * col_tiles, SMS, flatten=True):
        group_start = (tile // group_tiles) * group_m
        group_rows = tl.minimum(row_tiles - group_start, group_m)
        in_group = tile % group_tiles
        row = (group_start + in_group % group_rows) * bm + tl.arange(0, bm)
        col = (in_group // group_rows) * bn + tl.arange(0, bn)
        if LARGE:
            row = row.to(tl.int64)
            col = col.to(tl.int64)
        # Redirect the tail to a valid row. Keep alignment hints for vectorized
        # loads even when the last output tile is partial.
        xrow = tl.max_contiguous(tl.multiple_of(tl.where(row < M, row, 0), bm), bm)
        acc = tl.full((bm, bn), 0, tl.float32)
        for start in range(0, K, bk):
            inner = start + tl.arange(0, bk)
            a = tl.load(x + xrow[:, None] * K + inner[None, :])
            b = tl.load(weight + inner[:, None] + col[None, :] * K)
            acc = tl.dot(a, b, acc)
        if HAS_BIAS:
            acc += tl.load(bias + col).to(tl.float32)[None, :]
        tl.store(output + row[:, None] * N + col[None, :], acc, row[:, None] < M)


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_fp32: bool = False,
    *,
    mlp_num_sms: int = 0,
) -> torch.Tensor:
    """Multiply BF16 activations by a BF16 weight, with an optional scalar or per-feature bias."""
    n, k = weight.shape
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.size(-1) != k
        or x.device != weight.device
    ):
        raise ValueError(
            "batch-invariant linear requires compatible BF16 input and weight"
        )
    if bias is not None:
        if bias.device != x.device:
            raise ValueError("batch-invariant linear requires bias on the input device")
        # The kernel reads one bias row shared by every input token.
        row_shape = (1,) * (x.ndim - 1) + (n,)
        bias = bias.expand(row_shape).reshape(n).contiguous()
    output = torch.empty(
        (*x.shape[:-1], n),
        dtype=torch.float32 if output_fp32 else torch.bfloat16,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    rows = x.reshape(-1, k).contiguous()
    m = rows.size(0)
    if (
        mlp_num_sms > 0
        and m >= 256
        and n % 128 == 0
        and k % 64 == 0
        and weight.is_contiguous()
    ):
        # Preserve the small-M path for decode. Both kernels use one FP32
        # accumulator per output and visit the same 64-wide K tiles in order.
        grid = (min(mlp_num_sms, triton.cdiv(m, 128) * (n // 128)),)
        _persistent_mlp_linear[grid](
            rows,
            weight,
            bias,
            output,
            m,
            n,
            k,
            mlp_num_sms,
            bias is not None,
            max(rows.numel(), weight.numel(), output.numel()) > 2**31,
            num_warps=8,
            num_stages=3,
        )
        return output
    bm, bn, bk = 16, 16 if n <= 16 else 64, 64
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        rows,
        weight,
        bias,
        output,
        m,
        n,
        k,
        *weight.stride(),
        bm,
        bn,
        bk,
        bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return output


class BatchInvariantLinear(CudaF16Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return batch_invariant_linear(input, self.weight, self.bias)


class BatchInvariantMLPLinear(BatchInvariantLinear):
    """Qwen3-8B gate/up or down projection with an output-major BF16 weight."""

    def __init__(self, linear: BatchInvariantLinear):
        # Keep a single packed storage. The model binds its loader's [K,N]
        # weight view to this same buffer for checkpoint updates.
        super().__init__(linear.weight.contiguous().T, bias=linear.bias)
        properties = torch.cuda.get_device_properties(self.weight.device)
        self.num_sms = properties.multi_processor_count

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return batch_invariant_linear(
            input, self.weight, self.bias, mlp_num_sms=self.num_sms
        )


def prepare_qwen3_batch_invariant_mlp(
    config: "ModelConfig", mlp: "DenseMLP", weights: dict[str, torch.Tensor]
) -> None:
    """Pack the 8B projections and bind the loader to the same storage."""
    from rtp_llm.models_py.determinism.batch_invariant_config import is_qwen3_8b_dense
    from rtp_llm.resident_weights import WeightDict
    from rtp_llm.utils.model_weight import W

    if not is_qwen3_8b_dense(config):
        return
    # Published resident slots own the loader tensors and their original
    # strides. Keep those same views for cold starts and attached slots.
    if isinstance(weights, WeightDict) and weights.manager is not None:
        return

    mlp.up_proj = BatchInvariantMLPLinear(mlp.up_proj)
    gate_up = mlp.up_proj.weight.T
    if W.ffn_w13 in weights:
        weights[W.ffn_w13] = gate_up
    else:
        weights[W.ffn_w1], weights[W.ffn_w3] = gate_up.split(
            [weights[W.ffn_w1].size(1), weights[W.ffn_w3].size(1)], dim=1
        )
    mlp.down_proj = BatchInvariantMLPLinear(mlp.down_proj)
    weights[W.ffn_w2] = mlp.down_proj.weight.T


@triton.jit
def _kv_step(
    q, key, value, table, kv_head, start, length, positions, maximum, total, acc,
    PAGE_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr, TOKEN_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr, TABLE_COL_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr,
    D: tl.constexpr, N: tl.constexpr, SCALE_LOG2: tl.constexpr,
    BOUNDARY: tl.constexpr,
):
    n = start + tl.arange(0, N)
    d = tl.arange(0, D)
    if PAGE_SIZE % N == 0:
        # The complete KV tile (including a partial final tile) is in one page.
        page = tl.load(table + (start // PAGE_SIZE) * TABLE_COL_STRIDE).to(tl.int64)
    else:
        page = tl.load(
            table + (n // PAGE_SIZE) * TABLE_COL_STRIDE,
            mask=(n < length) if BOUNDARY else True,
            other=0,
        ).to(tl.int64)
    offset = page * PAGE_STRIDE + kv_head * HEAD_STRIDE + (n % PAGE_SIZE) * TOKEN_STRIDE
    offset = offset[:, None] + d[None, :] * DIM_STRIDE
    if BOUNDARY:
        k = tl.load(key + offset, mask=(n < length)[:, None], other=0)
        v = tl.load(value + offset, mask=(n < length)[:, None], other=0)
    else:
        k = tl.load(key + offset)
        v = tl.load(value + offset)

    scores = tl.dot(q, tl.trans(k))
    if BOUNDARY:
        scores = tl.where(n[None, :] <= positions[:, None], scores, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(scores, axis=1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp2((maximum - safe_max) * SCALE_LOG2)
    p = tl.exp2(tl.fma(scores, SCALE_LOG2, -safe_max[:, None] * SCALE_LOG2))
    total = total * alpha + tl.sum(p, axis=1)
    acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
    return new_max, total, acc


@triton.jit
def _tiled_attention(
    Q, K, V, O, Table, CuQ, UsedK,
    Q_ROW: tl.constexpr, Q_HEAD: tl.constexpr, Q_DIM: tl.constexpr,
    PAGE_STRIDE: tl.constexpr, HEAD_STRIDE: tl.constexpr, TOKEN_STRIDE: tl.constexpr,
    DIM_STRIDE: tl.constexpr, TABLE_ROW: tl.constexpr, TABLE_COL: tl.constexpr,
    CU_STRIDE: tl.constexpr, LENGTH_STRIDE: tl.constexpr, PAGE_SIZE: tl.constexpr,
    Q_HEADS: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
    SCALE_LOG2: tl.constexpr, DECODE: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    request = tl.program_id(2)
    row = tl.arange(0, M)
    dims = tl.arange(0, D)
    first_q = tl.load(CuQ + request * CU_STRIDE)
    q_len = tl.load(CuQ + (request + 1) * CU_STRIDE) - first_q
    length = tl.load(UsedK + request * LENGTH_STRIDE)
    if q_len == 0:
        return

    if DECODE:
        # Put the singleton queries sharing one KV head in different MMA rows.
        kv_head = head
        heads = kv_head * GROUP + row
        packed_rows = first_q + tl.full((M,), 0, tl.int32)
        positions = length - 1 + tl.full((M,), 0, tl.int32)
        valid = row < GROUP
        last_block = tl.maximum(length - 1, 0) // N
    else:
        # Absolute positions keep the row and KV tile of a query independent
        # of where the scheduler cuts a prefix or packs another request.
        prefix = length - q_len
        start = (prefix // M + tile) * M
        if start >= length:
            return
        positions = start + row
        packed_rows = first_q + positions - prefix
        valid = (positions >= prefix) & (positions < length)
        kv_head = head // GROUP
        heads = head + tl.full((M,), 0, tl.int32)
        last_block = start // N

    q_offsets = packed_rows.to(tl.int64) * Q_ROW + heads * Q_HEAD
    q = tl.load(
        Q + q_offsets[:, None] + dims[None, :] * Q_DIM, mask=valid[:, None], other=0
    )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    total = tl.full((M,), 0, tl.float32)
    acc = tl.full((M, D), 0, tl.float32)
    table = Table + request * TABLE_ROW
    if length > 0:
        # Q and KV tiles are the same width. Only the newest logical KV tile
        # intersects the causal diagonal. Every preceding tile is fully visible.
        maximum, total, acc = _kv_step(
            q, K, V, table, kv_head, last_block * N, length, positions, maximum,
            total, acc, PAGE_STRIDE, HEAD_STRIDE, TOKEN_STRIDE, DIM_STRIDE, TABLE_COL,
            PAGE_SIZE, D, N, SCALE_LOG2, True,
        )
        for i in range(last_block):
            start_k = (last_block - i - 1) * N
            maximum, total, acc = _kv_step(
                q, K, V, table, kv_head, start_k, length, positions, maximum,
                total, acc, PAGE_STRIDE, HEAD_STRIDE, TOKEN_STRIDE, DIM_STRIDE, TABLE_COL,
                PAGE_SIZE, D, N, SCALE_LOG2, False,
            )
    out_offsets = (packed_rows.to(tl.int64) * Q_HEADS + heads) * D
    result = acc / tl.where(total > 0, total, 1.0)[:, None]
    tl.store(O + out_offsets[:, None] + dims[None, :], result, mask=valid[:, None])


def tiled_paged_attention(
    q: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    *,
    decode: bool = False,
) -> torch.Tensor:
    """Attend packed queries to each request's cached prefix and causal suffix.

    The cache uses [page, K/V, KV head, token, dim]. Decode accepts one query
    per active request. All metadata values are read on the device at replay.
    """
    tokens, q_heads, dim = q.shape
    _, pair, kv_heads, page_size, cache_dim = cache.shape
    if q.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("tiled paged attention requires BF16 Q and KV")
    if pair != 2 or dim != cache_dim or dim < 16 or dim & (dim - 1):
        raise ValueError("invalid KV cache shape or attention head dimension")
    if kv_heads < 1 or q_heads % kv_heads or page_size < 1:
        raise ValueError("invalid grouped-query head count or KV page size")
    group = q_heads // kv_heads
    if decode and (max_seqlen_q != 1 or group > 64):
        raise ValueError("grouped decode requires singleton queries and <=64 heads per KV head")
    batch = seqused_k.numel()
    if block_table.ndim != 2 or block_table.size(0) != batch:
        raise ValueError("expected one block-table row per request")
    if cu_seqlens_q.shape != (batch + 1,) or seqused_k.ndim != 1:
        raise ValueError("expected packed query offsets and one KV length per request")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if tokens == 0:
        return output

    tile = 64
    # An unaligned prefix can require one more tile than ceil(q_len / tile).
    q_tiles = 1 if decode else triton.cdiv(max_seqlen_q + tile - 1, tile)
    grid = (q_tiles, kv_heads if decode else q_heads, batch)
    _tiled_attention[grid](
        q, cache[:, 0], cache[:, 1], output, block_table, cu_seqlens_q, seqused_k,
        *q.stride(), cache.stride(0), cache.stride(2), cache.stride(3), cache.stride(4),
        *block_table.stride(), cu_seqlens_q.stride(0), seqused_k.stride(0),
        page_size, q_heads, group, dim, dim**-0.5 * 1.4426950408889634,
        decode, tile, tile, num_warps=4, num_stages=2,
    )
    return output


class BatchInvariantAttention(FMHAImplBase):
    """Apply paged attention after RTP's fused RoPE and KV write."""

    def __init__(self, config: AttentionConfigs, inputs: PyAttentionInputs):
        self.config = config
        self.is_prefill = inputs.is_prefill
        self.block_table = inputs.kv_cache_kernel_block_id_device
        self.paged_attention = tiled_paged_attention
        if inputs.is_prefill:
            self.rope = FusedRopeKVCachePrefillOpQOut(config)
            self.cu_seqlens_q = inputs.cu_seqlens_device
            self.kv_lengths = inputs.input_lengths_device + inputs.prefix_lengths_device
            self.max_q = int(inputs.input_lengths.max())
        else:
            self.rope = FusedRopeKVCacheDecodeOp(config)
            self.kv_lengths = inputs.sequence_lengths_plus_1_device
            self.cu_seqlens_q = inputs.decode_cu_seqlens_device
            self.max_q = 1
        self.rope_params = self.rope.prepare(inputs)

    @staticmethod
    def support(config: AttentionConfigs, inputs: PyAttentionInputs) -> bool:
        return (
            config.is_causal
            and config.need_rope_kv_cache
            and not config.use_mla
            and config.kv_cache_dtype == KvCacheDataType.BASE
            and not inputs.is_target_verify
        )

    def support_cuda_graph(self) -> bool:
        return not self.is_prefill

    def prepare_cuda_graph(self, inputs: PyAttentionInputs) -> None:
        # The runner refreshes the page table and KV lengths in place. RoPE
        # also owns a position buffer and derives a separate KV offset table.
        params = self.rope.prepare(inputs, forbid_reallocation=True)
        copy_kv_cache_offset(self.rope_params.kv_cache_offset, params.kv_cache_offset)

    def forward(
        self, qkv: torch.Tensor, kv_cache: Optional[LayerKVCache], layer_idx: int = 0
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("batch-invariant attention requires paged KV cache")
        config = self.config
        q = self.rope.forward(qkv, kv_cache, self.rope_params)
        q = q.reshape(-1, config.head_num, config.size_per_head)
        cache = reshape_paged_kv_cache(
            kv_cache.kv_cache_base,
            config.kv_head_num,
            config.kernel_tokens_per_block,
            config.size_per_head,
        )
        return self.paged_attention(
            q,
            cache,
            self.block_table,
            self.cu_seqlens_q,
            self.kv_lengths,
            self.max_q,
            decode=not self.is_prefill,
        )


def create_batch_invariant_impl(
    attn_configs: AttentionConfigs,
    attn_inputs: PyAttentionInputs,
    is_cuda_graph: bool,
    cuda_graph_selection_mode: Optional[str],
) -> BatchInvariantAttention:
    if get_device_type() != DeviceType.Cuda:
        raise ValueError("batch-invariant attention requires a CUDA backend")
    if not BatchInvariantAttention.support(attn_configs, attn_inputs):
        raise ValueError("batch-invariant attention requires causal BF16 GQA")
    if (
        is_cuda_graph
        and cuda_graph_selection_mode == CudaGraphSelectionMode.GENERATION_PREFILL_GRAPH
    ):
        raise GenerationPrefillCudaGraphUnsupportedBackend(
            "batch-invariant attention supports pure decode CUDA Graphs"
        )
    expected_mode = (
        CudaGraphSelectionMode.DECODE_GRAPH
        if is_cuda_graph
        else CudaGraphSelectionMode.EAGER
    )
    if cuda_graph_selection_mode not in (None, expected_mode):
        raise ValueError("inconsistent batch-invariant CUDA graph selection mode")
    if is_cuda_graph and attn_inputs.is_prefill:
        raise ValueError("batch-invariant CUDA graph requires pure decode")
    attn_inputs.is_cuda_graph = is_cuda_graph
    return BatchInvariantAttention(attn_configs, attn_inputs)


# Each route owns one output row. Keep the same unsplit K traversal in both
# projections regardless of other tokens selecting this expert.
_MOE_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 4,
}
_EXPERT_WORKSPACE_BYTES = 64 * 1024 * 1024


def batch_invariant_moe_plan(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[dict[str, int], int]:
    """Return the fixed GEMM tile and the maximum token count per projection chunk."""
    rows, hidden = hidden_states.shape
    experts, _, intermediate = w2.shape
    if (
        hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or min(hidden, intermediate) <= 0
        or w2.size(1) != hidden
        or w1.shape != (experts, 2 * intermediate, hidden)
    ):
        raise ValueError("batch-invariant MoE requires compatible BF16 expert weights")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.size(0) != rows
        or not 0 < topk_ids.size(1) <= experts
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype != torch.float32
        or any(
            t.device != hidden_states.device for t in (w1, w2, topk_ids, topk_weights)
        )
    ):
        raise ValueError(
            "batch-invariant MoE requires matching expert IDs and FP32 routing weights"
        )

    top_k = topk_ids.size(1)
    # At peak the activation is live alongside either the up/gate or the down
    # projection. Bound these route buffers even for a long prefill.
    bytes_per_row = (
        top_k
        * (intermediate + max(2 * intermediate, hidden))
        * hidden_states.element_size()
    )
    return _MOE_GEMM_CONFIG, max(1, _EXPERT_WORKSPACE_BYTES // bytes_per_row)


def batch_invariant_experts(
    hidden_states, w1, w2, topk_ids, topk_weights, project_experts
) -> torch.Tensor:
    """Project bounded chunks with fixed tiles and sum in routing-slot order."""
    from rtp_llm.models_py.triton_kernels.moe.fixed_order_route_reduce import (
        fixed_order_fp32_route_reduce,
    )

    tile, chunk_size = batch_invariant_moe_plan(
        hidden_states, w1, w2, topk_ids, topk_weights
    )
    output = hidden_states.new_empty(hidden_states.shape)
    with torch.cuda.device_of(hidden_states):
        for start in range(0, hidden_states.size(0), chunk_size):
            chunk = slice(start, start + chunk_size)
            routes = project_experts(
                hidden_states[chunk],
                topk_ids[chunk].contiguous(),
                topk_weights[chunk].contiguous(),
                tile,
                tile,
            )
            fixed_order_fp32_route_reduce(routes, output[chunk], topk_ids.size(1))
            del routes  # Release this chunk before projecting the next.
    return output
