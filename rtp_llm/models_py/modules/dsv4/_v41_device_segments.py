"""Device-side MoE segment planning with fixed capacity (G5 / R5-c).

The production contiguous routed layout reads per-expert counts back to the
host (one blocking DtoH per MoE layer per chunk) to size allocations and
build per-row expert ids. This module replaces both with static-capacity
device tensors, removing the readback and the graph-capture blocker
(v41_moe.py "contiguous routing uses eager host counts").

Capacity bound: for alignment A and per-expert counts c_e with
sum(c_e) <= route_count,

    sum_e align_A(c_e) <= route_count + E*(A-1) <= capacity_for(...) - A

so the final A-row block is always tail (layout -1) and serves as the sink
for inactive routes. DeepGEMM 2.6.1 skips m_indices == -1 tiles in the
contiguous grouped layout (sm120_fp8_fp4_gemm_1d1d.cuh), so tail and sink
rows cost nothing and are never gathered.

Flag: DSV41_MOE_DEVICE_SEGMENTS=1 (default off; host path untouched).
"""

import os
import sys

import torch

DEVICE_SEGMENTS_FLAG = "DSV41_MOE_DEVICE_SEGMENTS"
_DSEG_ANNOUNCED = False


def device_segments_enabled() -> bool:
    return os.environ.get(DEVICE_SEGMENTS_FLAG, "0") == "1"


def announce_device_segments(detail: str = "") -> None:
    """One-shot [DSEGM] engagement proof (R18/C): the flag alone does not
    prove the device branch ran (it also requires the deepgemm backend and
    the contiguous layout)."""
    global _DSEG_ANNOUNCED
    if not _DSEG_ANNOUNCED:
        _DSEG_ANNOUNCED = True
        print(
            f"[DSEGM] device-side MoE segments engaged — no host count readback{detail}",
            file=sys.stderr,
            flush=True,
        )


def capacity_for(route_count: int, experts: int, align: int) -> int:
    """Static aligned capacity: worst-case padded segments plus a sink block."""
    if route_count < 0 or experts < 1 or align < 1:
        raise ValueError("V4.1 device segments require positive geometry")
    bound = route_count + experts * (align - 1)
    return ((bound + align - 1) // align) * align + align


def device_segment_plan(
    counts: torch.Tensor,
    sorted_experts: torch.Tensor,
    offsets: torch.Tensor,
    experts: int,
    align: int,
    capacity: int,
    sink: int,
):
    """Return (destinations, layout, active_sorted) computed entirely on device.

    counts: [experts+1] int64 (index experts = inactive sentinel bucket)
    sorted_experts: [route_count] expert key per sorted route (sentinel for
        inactive), in the stable-argsort order of the caller
    offsets: [route_count] position of each sorted route inside its expert's
        key run (already computed by the caller for both layouts)
    destinations: [route_count] int64 row for each sorted route; inactive
        routes share the sink row (written but never read)
    layout: [capacity] int32 per-row expert id; -1 on pad rows inside the
        tail region and on the sink block
    """
    active_sorted = sorted_experts < experts
    aligned = (counts[:experts] + (align - 1)) // align * align
    seg_ends = aligned.cumsum(0)
    seg_starts = seg_ends - aligned
    total = seg_ends[experts - 1]
    dest_active = (
        seg_starts.index_select(0, sorted_experts.clamp(max=experts - 1)) + offsets
    )
    destinations = torch.where(
        active_sorted, dest_active, dest_active.new_full((), sink)
    )
    blocks = torch.arange(0, capacity, align, device=counts.device, dtype=torch.int64)
    owner = torch.searchsorted(seg_ends, blocks, right=True)
    layout_blocks = torch.where(blocks < total, owner, owner.new_full((), -1))
    layout = layout_blocks.to(torch.int32).repeat_interleave(align)
    return destinations, layout, active_sorted
