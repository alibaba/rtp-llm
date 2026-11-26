"""Batch-invariant BF16 operators following vLLM's fixed reduction order.

Linear traverses K in fixed tiles. Dense paged attention uses one unsplit
online softmax per query and KV head. TP all-reduce sums in ascending group-rank order.
MoE uses fixed expert tiles and sums weighted outputs in routing order.
Qwen4-Max M11 uses fixed per-head reductions, ordered routing and eight QSA splits.
"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

_ALL_REDUCE_WORKSPACE_BYTES = 16 * 1024 * 1024


@triton.jit(do_not_specialize=["count"])
def _fixed_rank_sum(
    gathered, output, count, WORLD_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < count
    acc = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(WORLD_SIZE):
        values = tl.load(gathered + rank * count + offsets, live, other=0)
        acc += values.to(tl.float32)
    tl.store(output + offsets, acc, live)


def _all_reduce_chunks(source: torch.Tensor, dest: torch.Tensor, limit: int):
    """Slice matching logical rows without flattening a strided tensor into a full copy."""
    if source.numel() <= limit:
        yield source, dest
        return
    row_size = source.numel() // source.size(0)
    if row_size > limit:
        for row in range(source.size(0)):
            yield from _all_reduce_chunks(source[row], dest[row], limit)
    else:
        rows = limit // row_size
        for start in range(0, source.size(0), rows):
            yield source[start : start + rows], dest[start : start + rows]


def all_reduce_batch_invariant(
    tensor: torch.Tensor, group: dist.ProcessGroup, *, inplace: bool = True
) -> torch.Tensor:
    """Sum like-shaped BF16 tensors in TP group-rank order and round once per element."""
    if tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
        raise ValueError("batch-invariant all-reduce requires a strided BF16 tensor")
    if group is None:
        raise ValueError("batch-invariant all-reduce requires a TP process group")
    world_size = dist.get_world_size(group)
    if world_size < 1:
        raise ValueError(
            "batch-invariant all-reduce requires membership in the TP group"
        )
    if world_size == 1 or tensor.numel() == 0:
        return tensor if inplace else tensor.clone()

    if (
        inplace
        and not tensor.is_contiguous()
        and torch._debug_has_internal_overlap(tensor) == 1
    ):
        raise ValueError(
            "in-place batch-invariant all-reduce requires non-overlapping elements"
        )
    output = tensor if inplace else torch.empty_like(tensor)
    # Reserve space for one rank-major gather and at most one local staging
    # buffer. Chunk boundaries are independent of strides on individual ranks.
    capacity = max(
        1, _ALL_REDUCE_WORKSPACE_BYTES // ((world_size + 1) * tensor.element_size())
    )
    capacity = min(capacity, tensor.numel())
    gathered = tensor.new_empty(world_size * capacity)
    staging = None
    if not (tensor.is_contiguous() and output.is_contiguous()):
        staging = tensor.new_empty(capacity)

    with torch.cuda.device_of(tensor):
        for source, dest in _all_reduce_chunks(tensor, output, capacity):
            count = source.numel()
            if source.is_contiguous():
                send = source.view(-1)
            else:
                send = staging[:count]
                send.view_as(source).copy_(source)
            # Synchronous c10d semantics order NCCL and the caller's current
            # CUDA stream in both directions, including workspace reuse.
            dist.all_gather_into_tensor(
                gathered[: world_size * count], send, group=group
            )
            result = dest.view(-1) if dest.is_contiguous() else staging[:count]
            _fixed_rank_sum[(triton.cdiv(count, 256),)](
                gathered, result, count, WORLD_SIZE=world_size, BLOCK=256
            )
            if not dest.is_contiguous():
                dest.copy_(result.view_as(dest))
    return output


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


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_fp32: bool = False,
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


@triton.jit
def _paged_attention(
    query,
    key,
    value,
    output,
    block_table,
    request_ids,
    kv_lengths,
    TABLE_STRIDE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    SCALE: tl.constexpr,
    TILE_KV: tl.constexpr,
    TILE_Q: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    request = tl.load(request_ids + token)
    length = tl.load(kv_lengths + token)

    group_heads = tl.arange(0, TILE_Q)
    heads = kv_head * QUERIES_PER_KV + group_heads
    dims = tl.arange(0, HEAD_SIZE)
    q = tl.load(
        query + token * Q_STRIDE + heads[:, None] * HEAD_SIZE + dims[None, :],
        group_heads[:, None] < QUERIES_PER_KV,
        other=0,
    )
    maximum = tl.full((TILE_Q,), float("-inf"), tl.float32)
    denominator = tl.full((TILE_Q,), 0, tl.float32)
    numerator = tl.full((TILE_Q, HEAD_SIZE), 0, tl.float32)

    for start in range(0, length, TILE_KV):
        positions = start + tl.arange(0, TILE_KV)
        live = positions < length
        pages = tl.load(
            block_table + request * TABLE_STRIDE + positions // PAGE_SIZE,
            live,
            other=0,
        ).to(tl.int64)
        offsets = (
            pages * PAGE_STRIDE
            + kv_head * HEAD_STRIDE
            + (positions % PAGE_SIZE) * TOKEN_STRIDE
        )
        k = tl.load(key + offsets[None, :] + dims[:, None], live[None, :], other=0)
        v = tl.load(value + offsets[:, None] + dims[None, :], live[:, None], other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(live[None, :], scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        weights = tl.exp2(scores - new_maximum[:, None])
        rescale = tl.exp2(maximum - new_maximum)
        numerator = tl.dot(weights.to(tl.bfloat16), v, numerator * rescale[:, None])
        denominator = denominator * rescale + tl.sum(weights, axis=1)
        maximum = new_maximum

    result = numerator / tl.where(denominator > 0, denominator, 1)[:, None]
    tl.store(
        output
        + token * (QUERIES_PER_KV * tl.num_programs(1) * HEAD_SIZE)
        + heads[:, None] * HEAD_SIZE
        + dims[None, :],
        result,
        group_heads[:, None] < QUERIES_PER_KV,
    )


def batch_invariant_paged_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    kv_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run BF16 GQA with a cache laid out as [pages, 2, kv_heads, page_size, dim]."""
    tokens, query_heads, head_size = query.shape
    _, _, kv_heads, page_size, _ = cache.shape
    if query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("batch-invariant attention requires BF16 queries and KV cache")
    if query.stride(-1) != 1 or query.stride(-2) != head_size:
        query = query.contiguous()
    query_per_kv = query_heads // kv_heads
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if tokens == 0:
        return output

    _paged_attention[(tokens, kv_heads)](
        query,
        cache[:, 0],
        cache[:, 1],
        output,
        block_table,
        request_ids,
        kv_lengths,
        block_table.stride(0),
        query.stride(0),
        cache.stride(0),
        cache.stride(2),
        cache.stride(3),
        page_size,
        head_size,
        query_per_kv,
        scale * 1.4426950408889634,
        TILE_KV=32,
        TILE_Q=max(16, triton.next_power_of_2(query_per_kv)),
        num_warps=4,
        num_stages=2,
    )
    return output


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


# Qwen4-Max M11. The model binds these operators to its existing layer classes.
def batch_invariant_gdn_l2norm(x: torch.Tensor) -> torch.Tensor:
    """Use the same per-head reduction on both sides of FLA's 128-row cutoff."""
    from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd_kernel1

    # GDN splits Q/K/V from a single projection. Its Q/K views can be strided.
    x = x.contiguous()
    output = torch.empty_like(x)
    width = x.size(-1)
    block = triton.next_power_of_2(width)
    if block > 65536 // x.element_size():
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")
    with torch.cuda.device_of(x):
        l2norm_fwd_kernel1[(x.numel() // width,)](
            x, output, D=width, BD=block, eps=1e-6, num_warps=8, num_stages=3
        )
    return output


def batch_invariant_chunk_gated_delta_rule(
    q, k, v, g, beta, *, use_qk_l2norm_in_kernel=False, **kwargs
):
    from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule

    if use_qk_l2norm_in_kernel:
        q = batch_invariant_gdn_l2norm(q)
        k = batch_invariant_gdn_l2norm(k)
    return chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)


def install_batch_invariant_gdn(attention):
    """Keep native GDN prefill and decode for either configured backend."""
    if os.environ.get("QWEN4_GDN_BACKEND", "native") not in ("native", "flashinfer"):
        raise ValueError("QWEN4_GDN_BACKEND must be native or flashinfer")
    attention.prefill_gdn.chunk_rule = batch_invariant_chunk_gated_delta_rule


class BatchInvariantQwen4MoeTopk(torch.nn.Module):
    """Use the ordered router for every batch size, including large prefills."""

    _EXPERT_COUNTS = (512, 768)

    def __init__(self, native):
        super().__init__()
        self.native = native
        config = native.config
        self.geometry_supported = (
            config.expert_num in self._EXPERT_COUNTS
            and config.moe_k == 10
            and bool(config.has_moe_norm)
        )

    def forward(self, router_logits, topk_ids, topk_weights):
        from rtp_llm.models_py.triton_kernels.moe.qwen4_topk import (
            _moe_topk,
            supports_topk_layout,
        )

        if router_logits.shape[0] == 0:
            return
        if not (
            self.geometry_supported
            and supports_topk_layout(router_logits, topk_ids, topk_weights)
            and router_logits.shape[1] in self._EXPERT_COUNTS
        ):
            raise RuntimeError("M11 batch-invariant MoE requires the ordered router")
        experts = router_logits.shape[1]
        _moe_topk[(router_logits.shape[0],)](
            router_logits,
            topk_ids,
            topk_weights,
            STRIDE=router_logits.stride(0),
            EXPERTS=experts,
            BLOCK=triton.next_power_of_2(experts),
            num_warps=4 if experts == 768 else 1,
            enable_fp_fusion=False,
        )


def grouped_rms_norm(x, gamma, group_size, eps):
    from rtp_llm.models_py.triton_kernels.gated_residual.norm import (
        grouped_rms_norm_triton,
    )

    return grouped_rms_norm_triton(
        x.contiguous(), gamma, group_size, eps, allow_fp32_gamma=True
    )


def head_rms_norm(x, gamma, eps):
    return grouped_rms_norm(x, gamma, x.shape[-1], eps)


class Qwen4QKRMSNormMixin:
    """Normalize the Q and K heads in place using the fixed per-head reduction."""

    def forward(self, hidden_states):
        assert hidden_states.dim() == 2
        q, k, _ = hidden_states.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = q.unflatten(-1, (self.head_num, self.size_per_head))
        k = k.unflatten(-1, (self.kv_head_num, self.size_per_head))
        q.copy_(head_rms_norm(q, self.q_weight, self.eps))
        k.copy_(head_rms_norm(k, self.k_weight, self.eps))
        return hidden_states


class Qwen4GatedResidualMixin:
    """Share fixed projections across standalone and fused injection/normalization."""

    def _norm(self, hyper_input):
        return grouped_rms_norm(
            hyper_input, self.norm_gamma, self.hidden_size, self.norm_eps
        )

    def _forward_normed(self, hyper_input, normed, *, down_input_scale=None):
        from rtp_llm.models_py.triton_kernels.gated_residual.down import (
            down_inject_epilogue,
            separate_down_inject_epilogue,
        )
        from rtp_llm.models_py.triton_kernels.gated_residual.mix import (
            try_fused_mix_reduce,
        )

        # Both representations use the same epilogue after the BF16 projection.
        inject_weights = None
        branches = self.hc_mult
        if self.merged_down_inject is not None:
            projected = batch_invariant_linear(normed, self.merged_down_inject)
            mix, inject_weights = down_inject_epilogue(
                projected, self.mix_down.shape[0], branches
            )
        else:
            down = batch_invariant_linear(normed, self.mix_down)
            injection = (
                batch_invariant_linear(normed, self.inject)
                if self.inject is not None
                else None
            )
            if branches == 4 and injection is not None:
                mix, inject_weights = down_inject_epilogue(
                    torch.cat((down, injection), dim=-1), down.shape[-1], branches
                )
            else:
                epilogue = (
                    separate_down_inject_epilogue(down) if branches == 4 else None
                )
                mix = epilogue[0] if epilogue is not None else F.silu(down / branches)
                if injection is not None:
                    inject_weights = 2 * torch.sigmoid(injection / branches)
        logits = batch_invariant_linear(mix, self.mix_up)
        mixed = try_fused_mix_reduce(logits, normed, branches)
        if mixed is None:
            shape = (branches, self.hidden_size)
            mixed = (
                logits.sigmoid().unflatten(-1, shape) * normed.unflatten(-1, shape)
            ).mean(-2)
        return mixed, hyper_input, inject_weights


def _is_m11_ple_input(states):
    return (
        states.is_cuda
        and torch.version.hip is None
        and states.ndim == 3
        and states.shape[-1] == 32768
        and torch.cuda.get_device_capability(states.device) == (10, 3)
    )


def ple_gated_values(layer, hidden, embeddings, padding_mask):
    """M11 uses one fused gate reduction per token and branch at every length."""
    from rtp_llm.models_py.kernels.cuda.qwen4_ple.gate import try_ple_gate

    values = None
    if _is_m11_ple_input(hidden):
        values = try_ple_gate(
            batch_invariant_linear(embeddings, layer.key_proj),
            batch_invariant_linear(embeddings, layer.value_proj),
            hidden.contiguous(),
            layer.norm_key,
            layer.norm_query,
            layer.norm_conv,
            layer.norm_eps,
            single_pass_triton=True,
        )
    if values is None:
        raise RuntimeError("M11 batch-invariant PLE requires the row-local gate")
    gated, normed = values
    if padding_mask is not None:
        keep = padding_mask.unsqueeze(-1).to(gated.dtype)
        return gated * keep, normed * keep
    return gated, normed


def ple_conv(layer, gated, normed, initial, residual=None):
    """Use the same 1024-element convolution tile for prefill and decode."""
    from rtp_llm.models_py.kernels.cuda.qwen4_ple.conv import try_ple_conv

    output = None
    if _is_m11_ple_input(gated):
        output = try_ple_conv(
            gated,
            normed,
            initial,
            layer.conv_weight,
            layer.conv_dilation,
            residual=residual.contiguous() if residual is not None else None,
            row_block_size=1024,
        )
    if output is None:
        raise RuntimeError(
            "M11 batch-invariant PLE requires the fixed-order convolution"
        )
    return output


def ple_page_snapshots(layer, hidden, embeddings, page_size, initial, add_residual):
    gated, normed = ple_gated_values(layer, hidden, embeddings, None)
    output = ple_conv(layer, gated, normed, initial, hidden if add_residual else None)
    batch, length, width = hidden.shape
    state_len = layer.short_conv_state_len
    if initial is None:
        initial = normed.new_zeros((batch, state_len, width))
    history = torch.cat((initial, normed), dim=1)
    ends = range(page_size, length, page_size)
    return output, [history[:, end : end + state_len] for end in ends] + [
        history[:, length : length + state_len]
    ]


class Qwen4QSAIndexerMixin:
    """Q/K projection for M11's paged indexer."""

    batch_invariant = True
    _allow_flashinfer_prefill = False

    def project_raw(self, hidden_states):
        return self.split_projected(batch_invariant_linear(hidden_states, self.qk_proj))

    def normalize_query(self, query):
        return head_rms_norm(query, self.q_norm_gamma, self.norm_eps)

    def project(self, hidden_states):
        query, keys = self.project_raw(hidden_states)
        return self.normalize_query(query), keys

    def select_paged_indices(
        self, hidden_states, runtime, rope_config, *, projected=None
    ):
        """Use the same paged indexer for cold, continued and decode requests."""
        if projected is None:
            query, raw_keys = self.project(hidden_states)
        else:
            query, raw_keys = projected
            query = self.normalize_query(query)

        if runtime.main_inputs.is_target_verify:
            select = runtime.select_target_verify_tokens
        elif runtime.main_inputs.is_prefill:
            select = runtime.select_prefix_reuse_prefill_tokens
        else:
            select = runtime.select_decode_tokens
        return select(query, raw_keys, indexer=self, rope_config=rope_config)

    @staticmethod
    def pool_completed_keys(keys, dtype, gamma, eps):
        """Normalize the four-key mean with a fixed reduction tree at any row count."""
        if keys.shape[1] != 4:
            raise ValueError("M11 batch-invariant QSA compresses four keys per block")
        pooled = ((keys[:, 0] + keys[:, 1]) + (keys[:, 2] + keys[:, 3])) * 0.25
        return head_rms_norm(pooled.to(dtype), gamma, eps)

    @staticmethod
    def topk_blocks(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Break ties by logical block ID, including when other rows widen scores."""
        blocks = scores.argsort(dim=-1, descending=True, stable=True)[:, :k]
        return scores.gather(1, blocks), blocks


def _prepare_grouped_splits(
    q,
    cache,
    selected,
    *,
    page_size,
    block_k,
    selected_has_count,
    graph_capture,
    output_gate,
    fp16_pv,
):
    """Pack M11 token IDs for the same eight grouped splits on every request."""
    from rtp_llm.models_py.kernels.cuda.qsa.qsa_topk_triton import compact_qsa_selection

    if (
        (q.shape[1], cache.shape[2], q.shape[3], page_size, block_k)
        != (64, 4, 256, 128, 64)
        or cache.dtype != torch.bfloat16
        or selected.shape[-1] != 2051
        or selected_has_count
        or graph_capture
        or output_gate is not None
        or fp16_pv
    ):
        raise ValueError("M11 batch-invariant QSA requires eager BF16 paged GQA")
    return compact_qsa_selection(selected), 8


def paged_attention(query, cache, table, lengths, selected, **kwargs):
    """Validate paged KV operands before launching the fixed M11 split kernel."""
    from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import (
        sparse_paged_gqa_attn,
    )

    return sparse_paged_gqa_attn(
        query,
        cache,
        table,
        lengths,
        selected,
        _prepare_grouped_splits=_prepare_grouped_splits,
        **kwargs,
    )


def ragged_paged_attention(
    q, cache, plan, *, heads, kv_heads, head_dim, trusted_selected
):
    """Bound the eight FP32 partials per query/head independently of the batch."""
    if q.shape[0] == 0:
        return q.new_empty(q.shape)
    outputs = []
    for index, request in enumerate(plan.requests):
        start, end = int(request.start), int(request.end)
        table = plan.block_table[index : index + 1].contiguous()
        for row in range(start, end, 128):
            stop = min(row + 128, end)
            columns = slice(row - start, stop - start)
            query = q[row:stop].view(1, stop - row, heads, head_dim)
            chunk = paged_attention(
                query.transpose(1, 2).contiguous(),
                cache,
                table,
                request.kv_lens[:, columns],
                request.selected_indices[:, columns],
                page_size=int(plan.page_size),
                kv_head_num=kv_heads,
                validate_selected=not trusted_selected,
            )
            outputs.append(chunk.transpose(1, 2).reshape(stop - row, heads * head_dim))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
