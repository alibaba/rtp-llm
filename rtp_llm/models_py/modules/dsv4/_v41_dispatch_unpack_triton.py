"""R34 item (dispatch pack/unpack chain): fused unpack of the EP dispatch payload.

The production unpack (V41DispatchLayout.unpack) does four separate strided
``.contiguous()`` D2D copies (values fp8 / packed scales / weights fp32 /
indices int32) from the packed uint8 payload — four kernel launches per EP
receive, per layer, per chunk. This fused kernel reads the payload once and
writes all four contiguous fields in a single launch.

BYTE-EXACT: pure data movement (verbatim byte copies), no arithmetic. The
output tensors are bit-identical to the unfused path (same bytes, contiguous).
"""

import os
import sys

import torch
import triton
import triton.language as tl

UNPACK_FUSE_FLAG = "DSV41_DISPATCH_UNPACK_FUSE"
_ANNOUNCED = [False]


def unpack_fuse_enabled() -> bool:
    if os.environ.get(UNPACK_FUSE_FLAG, "0") == "1":
        if not _ANNOUNCED[0]:
            _ANNOUNCED[0] = True
            print(
                f"[DISPATCHUNPACKFUSE] engaged ({UNPACK_FUSE_FLAG}=1): fused payload unpack",
                file=sys.stderr,
                flush=True,
            )
        return True
    return False


@triton.jit
def _fused_unpack_kernel(
    PAYLOAD,  # uint8 [rows, payload_bytes]
    VALUES,  # uint8 [rows, hidden]   (fp8 bytes)
    PACKED,  # uint8 [rows, packed_bytes]
    WEIGHTS,  # uint8 [rows, topk*4]   (fp32 bytes)
    INDICES,  # uint8 [rows, topk*4]   (int32 bytes)
    ROWS,
    HIDDEN: tl.constexpr,
    PACKED_BYTES: tl.constexpr,
    TOPKB: tl.constexpr,  # topk * 4 bytes
    PACKED_POW2: tl.constexpr,
    TOPKB_POW2: tl.constexpr,
    WEIGHT_OFF: tl.constexpr,
    INDEX_OFF: tl.constexpr,
    PAYLOAD_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= ROWS:
        return
    src = PAYLOAD + row * PAYLOAD_BYTES
    # values: [0, HIDDEN) bytes
    dv = VALUES + row * HIDDEN
    for off in range(0, HIDDEN, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < HIDDEN
        tl.store(dv + cols, tl.load(src + cols, mask=m), mask=m)
    # packed scales: [HIDDEN, HIDDEN+PACKED_BYTES) bytes (PACKED_POW2 >= PACKED_BYTES)
    pc = tl.arange(0, PACKED_POW2)
    pm = pc < PACKED_BYTES
    tl.store(
        PACKED + row * PACKED_BYTES + pc, tl.load(src + HIDDEN + pc, mask=pm), mask=pm
    )
    # weights: [WEIGHT_OFF, WEIGHT_OFF+TOPKB) bytes
    wb = tl.arange(0, TOPKB_POW2)
    wm = wb < TOPKB
    tl.store(
        WEIGHTS + row * TOPKB + wb, tl.load(src + WEIGHT_OFF + wb, mask=wm), mask=wm
    )
    # indices: [INDEX_OFF, INDEX_OFF+TOPKB) bytes
    ib = tl.arange(0, TOPKB_POW2)
    im = ib < TOPKB
    tl.store(
        INDICES + row * TOPKB + ib, tl.load(src + INDEX_OFF + ib, mask=im), mask=im
    )


def fused_unpack(payload, layout):
    """One launch replacing unpack's four .contiguous() copies. Byte-exact."""
    rows = payload.shape[0]
    hidden = layout.hidden
    topk = layout.topk
    packed_bytes = layout.hidden // 32
    weight_off = layout.weight_offset
    index_off = layout.index_offset
    payload_bytes = layout.payload_bytes
    dev = payload.device
    values = torch.empty((rows, hidden), dtype=torch.uint8, device=dev)
    packed = torch.empty((rows, packed_bytes), dtype=torch.uint8, device=dev)
    weights = torch.empty((rows, topk), dtype=torch.float32, device=dev)
    indices = torch.empty((rows, topk), dtype=torch.int32, device=dev)

    def _np2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    _fused_unpack_kernel[(rows,)](
        payload,
        values,
        packed,
        weights.view(torch.uint8),
        indices.view(torch.uint8),
        rows,
        hidden,
        packed_bytes,
        topk * 4,
        _np2(packed_bytes),
        _np2(topk * 4),
        weight_off,
        index_off,
        payload_bytes,
        1024,
    )
    return values.view(torch.float8_e4m3fn), packed, weights, indices
