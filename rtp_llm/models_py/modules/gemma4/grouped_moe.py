"""Device layout adapter for the installed DeepGEMM BF16 grouped core.

DeepGEMM's prefix sums delimit actual rows, but the start of every expert is
aligned. RTP's routing offsets have no padding. Keep that difference explicit;
passing the original offsets directly can underflow the grouped scheduler.
The existing BF16 GEMM, activation and combine rounding remain unchanged.
"""

import os
from typing import Callable, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_layout(
    OFFSETS, STARTS, ENDS, ALIGN: tl.constexpr, E: tl.constexpr, BLOCK: tl.constexpr
):
    expert = tl.arange(0, BLOCK)
    end = tl.load(OFFSETS + expert, expert < E, other=0)
    previous = tl.load(OFFSETS + expert - 1, (expert > 0) & (expert < E), other=0)
    count = tl.where(expert < E, end - previous, 0)
    span = tl.cdiv(count, ALIGN) * ALIGN
    start = tl.cumsum(span) - span
    tl.store(STARTS + expert, start, expert < E)
    tl.store(ENDS + expert, start + count, expert < E)


@triton.jit
def _pad_rows(
    X,
    OFFSETS,
    STARTS,
    ENDS,
    PADDED,
    R: tl.constexpr,
    CAPACITY: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    column = tl.program_id(1) * BK + tl.arange(0, BK)
    # The rounded starts are monotonic, including empty experts. Select the
    # rightmost start <= row; an empty group contributes no valid rows.
    lo = tl.full((BR,), 0, tl.int32)
    hi = tl.full((BR,), E, tl.int32)
    for _ in tl.static_range(8):
        mid = (lo + hi) // 2
        start = tl.load(STARTS + mid, mid < E, other=2147483647)
        take = start <= row
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    expert = tl.minimum(lo - 1, E - 1)
    start = tl.load(STARTS + expert)
    end = tl.load(ENDS + expert)
    original_start = tl.load(OFFSETS + expert - 1, expert > 0, other=0)
    original_row = original_start + row - start
    valid = (row < end) & (original_row < R) & (row < CAPACITY)
    value = tl.load(
        X + original_row[:, None].to(tl.int64) * K + column[None, :],
        valid[:, None] & (column[None, :] < K),
        other=0,
    )
    tl.store(
        PADDED + row[:, None].to(tl.int64) * K + column[None, :],
        value,
        (row[:, None] < CAPACITY) & (column[None, :] < K),
    )


@triton.jit
def _unpad_rows(
    PADDED,
    OFFSETS,
    STARTS,
    Y,
    R: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    query_row = tl.minimum(row, R - 1)
    column = tl.program_id(1) * BN + tl.arange(0, BN)
    lo = tl.full((BR,), 0, tl.int32)
    hi = tl.full((BR,), E, tl.int32)
    for _ in tl.static_range(8):
        mid = (lo + hi) // 2
        end = tl.load(OFFSETS + mid, mid < E, other=2147483647)
        take = end <= query_row
        lo = tl.where(take, mid + 1, lo)
        hi = tl.where(take, hi, mid)
    expert = lo
    original_start = tl.load(OFFSETS + expert - 1, expert > 0, other=0)
    padded_row = tl.load(STARTS + expert) + query_row - original_start
    mask = (row[:, None] < R) & (column[None, :] < N)
    value = tl.load(
        PADDED + padded_row[:, None].to(tl.int64) * N + column[None, :],
        mask,
        other=0,
    )
    tl.store(Y + row[:, None].to(tl.int64) * N + column[None, :], value, mask)


@triton.jit
def _aligned_gather(
    X,
    IDS,
    PERM,
    OFFSETS,
    STARTS,
    PADDED,
    R: tl.constexpr,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BR: tl.constexpr,
    BH: tl.constexpr,
):
    sorted_row = tl.program_id(0) * BR + tl.arange(0, BR)
    column = tl.program_id(1) * BH + tl.arange(0, BH)
    valid_row = sorted_row < R
    slot = tl.load(PERM + sorted_row, valid_row, other=0)
    expert = tl.load(IDS + slot, valid_row, other=0)
    original_start = tl.load(OFFSETS + expert - 1, valid_row & (expert > 0), other=0)
    padded_row = tl.load(STARTS + expert) + sorted_row - original_start
    token = slot // TOP_K
    mask = valid_row[:, None] & (column[None, :] < H)
    value = tl.load(
        X + token[:, None].to(tl.int64) * H + column[None, :], mask, other=0
    )
    tl.store(
        PADDED + padded_row[:, None].to(tl.int64) * H + column[None, :], value, mask
    )


@triton.jit
def _weighted_route(
    PADDED,
    IDS,
    INVERSE,
    WEIGHTS,
    OFFSETS,
    STARTS,
    slot,
    column,
    valid_row,
    H: tl.constexpr,
):
    expert = tl.load(IDS + slot, valid_row, other=0)
    sorted_row = tl.load(INVERSE + slot, valid_row, other=0)
    original_start = tl.load(OFFSETS + expert - 1, valid_row & (expert > 0), other=0)
    padded_row = tl.load(STARTS + expert) + sorted_row - original_start
    weight = tl.load(WEIGHTS + sorted_row, valid_row, other=0).to(tl.float32)
    value = tl.load(
        PADDED + padded_row[:, None].to(tl.int64) * H + column[None, :],
        valid_row[:, None] & (column[None, :] < H),
        other=0,
    ).to(tl.float32)
    # Match native WeightedReorder: FP32 multiply -> BF16 RN, before summing.
    return (value * weight[:, None]).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _weighted_top8_combine(
    PADDED,
    IDS,
    INVERSE,
    WEIGHTS,
    OFFSETS,
    STARTS,
    OUTPUT,
    M: tl.constexpr,
    H: tl.constexpr,
    BR: tl.constexpr,
    BH: tl.constexpr,
):
    token = tl.program_id(0) * BR + tl.arange(0, BR)
    column = tl.program_id(1) * BH + tl.arange(0, BH)
    valid_row = token < M
    slot = token * 8
    # Match gemma4Top8SumBf16Kernel exactly: four (i,i+4) pairs,
    # then ((pair0+pair1)+pair2)+pair3, then one BF16 RN conversion.
    p0 = _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot, column, valid_row, H
    ) + _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 4, column, valid_row, H
    )
    p1 = _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 1, column, valid_row, H
    ) + _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 5, column, valid_row, H
    )
    p2 = _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 2, column, valid_row, H
    ) + _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 6, column, valid_row, H
    )
    p3 = _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 3, column, valid_row, H
    ) + _weighted_route(
        PADDED, IDS, INVERSE, WEIGHTS, OFFSETS, STARTS, slot + 7, column, valid_row, H
    )
    result = ((p0 + p1) + p2) + p3
    tl.store(
        OUTPUT + token[:, None].to(tl.int64) * H + column[None, :],
        result,
        valid_row[:, None] & (column[None, :] < H),
    )


class DeepGemmGroupedMM:
    """Reuse the existing SM100 BF16 core with graph-mutable device offsets."""

    def __init__(self, module, alignment: int):
        self._module = module
        self.alignment = alignment

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        rows, reduction = x.shape
        experts, output_width, _ = weight.shape
        if rows == 0:
            return torch.empty((0, output_width), dtype=x.dtype, device=x.device)
        if not (
            x.is_contiguous()
            and x.dtype == torch.bfloat16
            and weight.is_contiguous()
            and weight.dtype == x.dtype
            and weight.device == x.device
            and offsets.device == x.device
            and offsets.dtype == torch.int32
            and offsets.is_contiguous()
            and offsets.shape == (experts,)
            and weight.shape[2] == reduction
            and 0 < experts <= 128
        ):
            raise ValueError(
                "DeepGEMM grouped adapter requires contiguous CUDA BF16 operands and int32 offsets"
            )
        starts = torch.empty_like(offsets)
        ends = torch.empty_like(offsets)
        _prepare_layout[(1,)](
            offsets,
            starts,
            ends,
            self.alignment,
            experts,
            triton.next_power_of_2(experts),
        )
        # At most min(E,R) groups are nonempty. Each extra nonempty group adds
        # at most one alignment tile beyond ceil(R/alignment).
        capacity = (
            triton.cdiv(rows, self.alignment) + min(experts, rows) - 1
        ) * self.alignment
        padded_x = torch.empty((capacity, reduction), dtype=x.dtype, device=x.device)
        padded_y = torch.empty((capacity, output_width), dtype=x.dtype, device=x.device)
        output = torch.empty((rows, output_width), dtype=x.dtype, device=x.device)
        _pad_rows[(triton.cdiv(capacity, 16), triton.cdiv(reduction, 256))](
            x,
            offsets,
            starts,
            ends,
            padded_x,
            rows,
            capacity,
            reduction,
            experts,
            16,
            256,
        )
        self._module.m_grouped_bf16_gemm_nt_contiguous(
            padded_x,
            weight,
            padded_y,
            ends,
            use_psum_layout=True,
            ensure_zero_padding=False,
        )
        _unpad_rows[(triton.cdiv(rows, 16), triton.cdiv(output_width, 256))](
            padded_y, offsets, starts, output, rows, output_width, experts, 16, 256
        )
        return output

    def forward_moe(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        offsets: torch.Tensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Pad once; preserve both BF16 GEMM and original GeGLU boundaries."""
        rows, hidden = x.shape
        experts, gate_up_width, _ = w1.shape
        if rows == 0:
            return torch.empty_like(x)
        capacity = (
            triton.cdiv(rows, self.alignment) + min(experts, rows) - 1
        ) * self.alignment
        starts = torch.empty_like(offsets)
        ends = torch.empty_like(offsets)
        _prepare_layout[(1,)](
            offsets,
            starts,
            ends,
            self.alignment,
            experts,
            triton.next_power_of_2(experts),
        )
        padded_x = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
        padded_gate_up = torch.empty(
            (capacity, gate_up_width), dtype=x.dtype, device=x.device
        )
        _pad_rows[(triton.cdiv(capacity, 16), triton.cdiv(hidden, 256))](
            x, offsets, starts, ends, padded_x, rows, capacity, hidden, experts, 16, 256
        )
        self._module.m_grouped_bf16_gemm_nt_contiguous(
            padded_x,
            w1,
            padded_gate_up,
            ends,
            use_psum_layout=True,
            ensure_zero_padding=False,
        )
        # Same-stream launches are ordered; allow the graph allocator to reuse
        # scratch only after its consuming GEMM has been enqueued.
        del padded_x
        # This is the same row-local production activation. Unused padded
        # rows never enter router weighting or the output reduction.
        padded_activation = activation(padded_gate_up)
        del padded_gate_up
        padded_output = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
        self._module.m_grouped_bf16_gemm_nt_contiguous(
            padded_activation,
            w2,
            padded_output,
            ends,
            use_psum_layout=True,
            ensure_zero_padding=False,
        )
        del padded_activation
        output = torch.empty_like(x)
        _unpad_rows[(triton.cdiv(rows, 16), triton.cdiv(hidden, 256))](
            padded_output, offsets, starts, output, rows, hidden, experts, 16, 256
        )
        return output

    def forward_routed(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_ids: torch.Tensor,
        permutation: torch.Tensor,
        inverse: torch.Tensor,
        sorted_weights: torch.Tensor,
        offsets: torch.Tensor,
        activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Fuse ingress gather and final BF16 weighted/reduced row restoration."""
        tokens, hidden = x.shape
        rows = permutation.numel()
        experts, gate_up_width, _ = w1.shape
        if rows == 0:
            return torch.empty_like(x)
        capacity = (
            triton.cdiv(rows, self.alignment) + min(experts, rows) - 1
        ) * self.alignment
        starts, ends = torch.empty_like(offsets), torch.empty_like(offsets)
        _prepare_layout[(1,)](
            offsets,
            starts,
            ends,
            self.alignment,
            experts,
            triton.next_power_of_2(experts),
        )
        padded_x = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
        _aligned_gather[(triton.cdiv(rows, 16), triton.cdiv(hidden, 256))](
            x,
            expert_ids,
            permutation,
            offsets,
            starts,
            padded_x,
            rows,
            hidden,
            8,
            16,
            256,
        )
        padded_gate_up = torch.empty(
            (capacity, gate_up_width), dtype=x.dtype, device=x.device
        )
        self._module.m_grouped_bf16_gemm_nt_contiguous(
            padded_x,
            w1,
            padded_gate_up,
            ends,
            use_psum_layout=True,
            ensure_zero_padding=False,
        )
        del padded_x
        padded_activation = activation(padded_gate_up)
        del padded_gate_up
        padded_output = torch.empty((capacity, hidden), dtype=x.dtype, device=x.device)
        self._module.m_grouped_bf16_gemm_nt_contiguous(
            padded_activation,
            w2,
            padded_output,
            ends,
            use_psum_layout=True,
            ensure_zero_padding=False,
        )
        del padded_activation
        output = torch.empty_like(x)
        _weighted_top8_combine[(triton.cdiv(tokens, 4), triton.cdiv(hidden, 256))](
            padded_output,
            expert_ids,
            inverse,
            sorted_weights,
            offsets,
            starts,
            output,
            tokens,
            hidden,
            4,
            256,
            enable_fp_fusion=False,
        )
        return output


def create_grouped_mm(
    w1: torch.Tensor, w2: torch.Tensor
) -> Optional[DeepGemmGroupedMM]:
    """Explicit candidate rollout; nearest unsupported cases keep Torch."""
    if os.environ.get("RTP_LLM_GEMMA4_GROUPED_GEMM_BACKEND", "torch") != "deepgemm":
        return None
    if not (
        w1.is_cuda
        and torch.version.hip is None
        and w1.dtype == torch.bfloat16
        and w2.dtype == w1.dtype
        and w1.is_contiguous()
        and w2.is_contiguous()
        and w2.device == w1.device
        and tuple(w1.shape) == (128, 1408, 2816)
        and tuple(w2.shape) == (128, 2816, 704)
        and torch.cuda.get_device_capability(w1.device)[0] == 10
    ):
        return None
    try:
        import deep_gemm
    except ImportError:
        return None
    if not callable(getattr(deep_gemm, "m_grouped_bf16_gemm_nt_contiguous", None)):
        return None
    alignment = int(deep_gemm.get_mk_alignment_for_contiguous_layout())
    if alignment <= 0 or alignment & (alignment - 1):
        return None
    return DeepGemmGroupedMM(deep_gemm, alignment)
