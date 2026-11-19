"""Wire layout and fixed workspace geometry for SM120 decode."""

from dataclasses import dataclass
from math import prod

import torch

from rtp_llm.models_py.utils.math import align, ceil_div

_ALIGNMENT = 128
_MAX_KERNEL_INDEX = (1 << 31) - 1


def decode_payload_offsets(hidden: int, topk: int) -> tuple[int, int, int]:
    # E4M3 values, one UE8M0 byte per 32 values, FP32 weights, int32 IDs.
    scale_end = hidden + hidden // 32
    weight_end = scale_end + topk * 4
    return scale_end, weight_end, weight_end + topk * 4


@dataclass(frozen=True)
class Sm120DecodeMoeGeometry:
    rows_per_rank: int
    world: int
    hidden: int
    inter: int
    total_experts: int
    topk: int

    def __post_init__(self) -> None:
        if (
            self.rows_per_rank <= 0
            or self.world not in (4, 8)
            or self.hidden <= 0
            or self.hidden % _ALIGNMENT != 0
            or self.inter <= 0
            or self.inter % (4 * _ALIGNMENT) != 0
            or self.total_experts <= 0
            or self.total_experts % self.world != 0
            or not 0 < self.topk <= min(self.total_experts, 32)
        ):
            raise ValueError(f"Unsupported SM120 decode graph MoE geometry: {self}")
        # Scatter stores a flattened expert row in int32. Quantization,
        # routing and the provider's other kernel extents use int32 as well.
        if (
            max(
                self.expert_capacity * self.local_experts,
                self.rows * self.payload_cols,
                self.rows * self.hidden,
                self.rows * self.topk,
                self.expert_capacity
                * self.local_experts
                * max(self.hidden, 2 * self.inter),
            )
            > _MAX_KERNEL_INDEX
        ):
            raise ValueError(
                f"SM120 decode graph MoE exceeds int32 kernel extents: {self}"
            )

    @property
    def rows(self) -> int:
        return self.rows_per_rank * self.world

    @property
    def local_experts(self) -> int:
        return self.total_experts // self.world

    @property
    def expert_capacity(self) -> int:
        # The payload coalesces repeated IDs, so each row contributes at
        # most one assignment per expert even for a hash-routing table.
        return align(self.rows, _ALIGNMENT)

    @property
    def payload_cols(self) -> int:
        return decode_payload_offsets(self.hidden, self.topk)[2]

    @property
    def workspace_bytes(self) -> int:
        return sum(
            prod(shape) * dtype.itemsize
            for _, shape, dtype, _ in self.workspace_layout()
        )

    def workspace_layout(self):
        rows, hidden, inter = self.rows, self.hidden, self.inter
        experts, capacity = self.local_experts, self.expert_capacity
        # Packed scale storage is transposed after taking a contiguous prefix.
        # Narrowing its logical row dimension would retain the wrong strides.
        return (
            ("payload", (self.rows_per_rank, self.payload_cols), torch.uint8, False),
            ("gathered", (rows, self.payload_cols), torch.uint8, False),
            ("partial", (rows, hidden), torch.float32, False),
            ("returned", (rows, hidden), torch.float32, False),
            ("output", (self.rows_per_rank, hidden), torch.float32, False),
            ("expert_x", (experts, capacity, hidden), torch.float8_e4m3fn, False),
            (
                "expert_x_scale",
                (experts, ceil_div(hidden // _ALIGNMENT, 4), capacity),
                torch.int32,
                True,
            ),
            ("gate_up", (experts, capacity, 2 * inter), torch.bfloat16, False),
            ("down_in", (experts, capacity, inter), torch.float8_e4m3fn, False),
            (
                "down_in_scale",
                (experts, inter // (4 * _ALIGNMENT), capacity),
                torch.int32,
                True,
            ),
            ("down_out", (experts, capacity, hidden), torch.bfloat16, False),
            ("start_loc", (experts,), torch.int32, False),
            ("out_index", (rows, self.topk), torch.int32, False),
        )


def local_decode_geometry(config, rows: int) -> Sm120DecodeMoeGeometry:
    return Sm120DecodeMoeGeometry(
        rows,
        config.ep_size,
        config.dim,
        config.moe_inter_dim,
        config.n_routed_experts,
        config.moe_k,
    )


def decode_workspace_geometry(
    buckets, token_widths, world, hidden, inter, total_experts, topk
):
    """Size one model's scratch for its largest sequential graph invocation."""
    return Sm120DecodeMoeGeometry(
        max(buckets) * max(token_widths), world, hidden, inter, total_experts, topk
    )
