"""Pack BF16 decode input and routes directly into the PPU GC MoE workspace."""

import triton
import triton.language as tl


@triton.jit
def _pack_inputs(
    X,
    IDS,
    WEIGHTS,
    VALID,
    PACKED,
    SCALES,
    OUT_IDS,
    OUT_WEIGHTS,
    FP8,
    FP8_SCALES,
    H: tl.constexpr,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    X_ROW: tl.constexpr,
    X_COL: tl.constexpr,
    ID_ROW: tl.constexpr,
    ID_COL: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    WEIGHT_COL: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    VALID_ROW: tl.constexpr,
    FP8_BLOCK: tl.constexpr,
    FP4_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if FP8_BLOCK > 0 and tile == FP4_TILES:
        cols = tl.arange(0, FP8_BLOCK)
        values = tl.load(X + row * X_ROW + cols * X_COL, cols < H, 0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(values), 0), 1.0e-10)
        inverse = tl.div_rn(1.0, amax) * 448.0
        scaled = tl.minimum(tl.maximum(values * inverse, -448.0), 448.0)
        tl.store(FP8 + row * H + cols, scaled, cols < H)
        tl.store(FP8_SCALES + row, amax * (1.0 / 448.0))
    else:
        _pack_mxfp4(
            X,
            IDS,
            WEIGHTS,
            VALID,
            PACKED,
            SCALES,
            OUT_IDS,
            OUT_WEIGHTS,
            H,
            K,
            OUT_K,
            X_ROW,
            X_COL,
            ID_ROW,
            ID_COL,
            WEIGHT_ROW,
            WEIGHT_COL,
            BLOCK,
            ROUTE_BLOCK,
            VALID_ROW,
            row,
            tile,
        )


@triton.jit
def _pack_mxfp4(
    X,
    IDS,
    WEIGHTS,
    VALID,
    PACKED,
    SCALES,
    OUT_IDS,
    OUT_WEIGHTS,
    H: tl.constexpr,
    K: tl.constexpr,
    OUT_K: tl.constexpr,
    X_ROW: tl.constexpr,
    X_COL: tl.constexpr,
    ID_ROW: tl.constexpr,
    ID_COL: tl.constexpr,
    WEIGHT_ROW: tl.constexpr,
    WEIGHT_COL: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    VALID_ROW: tl.constexpr,
    row,
    tile,
):
    col = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * X_ROW + col * X_COL, col < H, other=0).to(tl.float32)
    groups = tl.reshape(x, (BLOCK // 32, 32))
    amax = tl.max(tl.abs(groups), axis=1)
    # Round the FP32 scale up to a power of two before E2M1 conversion.
    scale = (amax / 6.0).to(tl.uint32, bitcast=True)
    exponent = ((scale + 0x007FFFFF) & 0x7F800000) >> 23
    inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    values = tl.reshape(groups * inverse[:, None], (BLOCK // 2, 2))
    lo, hi = tl.split(values)
    packed = tl.inline_asm_elementwise(
        """
        {
            .reg .b8 bits;
            cvt.rn.satfinite.e2m1x2.f32 bits, $1, $2;
            mov.b32 $0, {bits, bits, bits, bits};
        }
        """,
        constraints="=r,f,f",
        args=[hi, lo],
        dtype=tl.uint8,
        is_pure=True,
        pack=1,
    )
    pair = tile * (BLOCK // 2) + tl.arange(0, BLOCK // 2)
    group = tile * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    tl.store(PACKED + row * (H // 2) + pair, packed, pair < H // 2)
    tl.store(SCALES + row * (H // 32) + group, exponent, group < H // 32)
    if tile == 0:
        slot = tl.arange(0, ROUTE_BLOCK)
        ids = tl.load(IDS + row * ID_ROW + slot * ID_COL, slot < K, other=-1)
        weights = tl.load(
            WEIGHTS + row * WEIGHT_ROW + slot * WEIGHT_COL, slot < K, other=0
        )
        if VALID_ROW is not None:
            live = tl.load(VALID + row * VALID_ROW)
            ids = tl.where(live, ids, -1)
            weights = tl.where(live, weights, 0)
        tl.store(OUT_IDS + row * OUT_K + slot, ids, slot < OUT_K)
        tl.store(OUT_WEIGHTS + row * OUT_K + slot, weights, slot < OUT_K)


def pack_ppu_gc_moe_inputs(x, ids, weights, workspace, *, valid=None):
    """Submit one kernel and write only the rows used by this request."""
    rows, hidden = x.shape
    topk = ids.shape[1]
    if (
        rows > workspace.num_max_tokens_per_rank
        or hidden != workspace.hidden
        or ids.shape != weights.shape
        or ids.shape[0] != rows
        or not 0 < topk <= workspace.num_topk
    ):
        raise ValueError("input and route shapes must fit the PPU GC MoE workspace")
    if rows:
        block = 512 if rows < 8 else 1024
        tiles = triton.cdiv(hidden, block)
        shared_x = getattr(workspace, "shared_x", None)
        shared_sf = getattr(workspace, "shared_x_sf", None)
        _pack_inputs[(rows, tiles + int(shared_x is not None))](
            x,
            ids,
            weights,
            valid,
            workspace.x,
            workspace.x_sf,
            workspace.topk_idx,
            workspace.topk_weights,
            shared_x,
            shared_sf,
            hidden,
            topk,
            workspace.num_topk,
            *x.stride(),
            *ids.stride(),
            *weights.stride(),
            block,
            triton.next_power_of_2(workspace.num_topk),
            valid.stride(0) if valid is not None else None,
            triton.next_power_of_2(hidden) if shared_x is not None else 0,
            tiles,
            num_warps=4,
        )
