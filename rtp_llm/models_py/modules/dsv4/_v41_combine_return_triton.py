"""R25/F1 (KDA-FUSION-20261030 F3): fused ordered combine + MXFP8 return pack.

Flag ``DSV41_COMBINE_RETURN_FUSED`` (default "1" since R27 adoption;
"0" = escape: the two-kernel path is
untouched). Under "1" the sender-side MoE return tail runs ONE Triton pass
over [rows, 5120] instead of two:

  before: _combine_kernel (BF16 down rows -> FP32 partial, 74.7 us)
          + _return (FP32 partial -> BF16 round -> group-32 E4M3/UE8M0 wire
          packet, 70.9 us)  == ~145.6 us + an 84 MB FP32 roundtrip/layer
  after:  _combine_return_kernel (~90-100 us, no partial materialization)

Byte-exact by construction:
  * the K-slot accumulation replays _combine_kernel exactly — same slot
    order (route order after the stable argsort), same ``add.rn.f32``
    elementwise chain (no FMA: enable_fp_fusion=False), same masked
    BF16->FP32 loads with other=0 (adding +0.0 is exact);
  * the quantization replays _return exactly — the FP32 accumulator goes
    through the same ``.to(bfloat16).to(float32)`` rounding BEFORE
    ``_quantize`` (the imported, unmodified jit function), same group-32
    tiling (BLOCK_G=8 x 32 columns == the 256-column blocks of both
    parents), same device_assert, same stores.
Grid/launch match the parents: (rows, cdiv(hidden//32, 8)), num_warps=4.

Proof path: r25_f1_bitwise.py (synthetic production-shape bitwise compare
vs combine_ordered_rows + pack_return(fused=True)) BEFORE any timing arm;
the parity run then proves it end-to-end on real weights via the sha.
"""

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4._v41_moe_combine_triton import _add_rn
from rtp_llm.models_py.modules.dsv4._v41_packet_triton import _quantize

COMBINE_RETURN_FUSED_FLAG = "DSV41_COMBINE_RETURN_FUSED"
_ANNOUNCED = [False]


def combine_return_fused_enabled() -> bool:
    # R27: default flipped to "1" (adopted; R25 evidence: stack -42.4 ms,
    # 3/3, byte-exact). Escape: DSV41_COMBINE_RETURN_FUSED=0.
    return os.environ.get(COMBINE_RETURN_FUSED_FLAG, "1") == "1"


@triton.jit
def _combine_return_kernel(
    DOWN,
    MAP,
    Out,
    D: tl.constexpr,
    K: tl.constexpr,
    MS0: tl.constexpr,
    MS1: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    col = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    # Ordered FP32 combine — identical slot sequence and add.rn chain as
    # _combine_kernel; the 2-D [BLOCK_G, 32] indexing covers the same 256
    # columns in the same element order as its 1-D BLOCK=256 grid.
    accumulated = tl.full((BLOCK_G, 32), 0, tl.float32)
    for slot in tl.static_range(K):
        destination = tl.load(MAP + row * MS0 + slot * MS1)
        value = tl.load(
            DOWN + tl.maximum(destination, 0) * D + col,
            mask=(destination >= 0) & (col < D),
            other=0,
        ).to(tl.float32)
        accumulated = _add_rn(accumulated, value)
    # Return pack — identical rounding chain as _return: BF16 round-trip
    # BEFORE the group quantization, same finite-input assert.
    x = accumulated.to(tl.bfloat16).to(tl.float32)
    tl.device_assert(
        tl.sum((tl.abs(x) < float("inf")).to(tl.int32), 1) == 32,
        "V4.1 activation quantization requires finite inputs",
    )
    values, encoded = _quantize(x)
    tl.store(Out + row * ROW_BYTES + col, values, col < D)
    tl.store(Out + row * ROW_BYTES + D + group, encoded.to(tl.uint8), group < D // 32)


def combine_return_fused(down, route_map, return_layout, *, _validated_map=False):
    """One-pass combine+pack. Returns the uint8 wire packet [rows,
    payload_bytes], or None when any guard fails (caller keeps the legacy
    two-kernel path — silent fallbacks are caught by the [COMBRET] engage
    marker at the parity gate)."""
    from rtp_llm.models_py.modules.dsv4.v41_return import V41ReturnLayout

    if not (
        type(return_layout) is V41ReturnLayout
        and return_layout.mode == "mxfp8"
        and down.device.type == "cuda"
        and torch.cuda.get_device_capability(down.device) == (12, 0)
        and down.ndim == 2
        and down.shape[0] >= 0
        and down.shape[1] == return_layout.hidden
        and down.dtype == torch.bfloat16
        and down.is_contiguous()
        and route_map.device == down.device
        and route_map.ndim == 2
        and 0 < route_map.shape[1] <= 32
        and route_map.dtype in (torch.int32, torch.int64)
    ):
        return None
    rows, topk = route_map.shape
    if not _validated_map:
        torch._assert_async(
            ((route_map >= -1) & (route_map < down.shape[0])).all(),
            "V4.1 fused combine-return route destination is outside expert storage",
        )
    output = torch.empty(
        (rows, return_layout.payload_bytes), dtype=torch.uint8, device=down.device
    )
    if not rows:
        return output
    if not _ANNOUNCED[0]:
        _ANNOUNCED[0] = True
        import sys

        print(
            f"[COMBRET] engaged ({COMBINE_RETURN_FUSED_FLAG}=1): fused ordered "
            "combine + MXFP8 return pack",
            file=sys.stderr,
            flush=True,
        )
    _combine_return_kernel[(rows, triton.cdiv(return_layout.hidden // 32, 8))](
        down,
        route_map,
        output,
        return_layout.hidden,
        topk,
        route_map.stride(0),
        route_map.stride(1),
        return_layout.payload_bytes,
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
