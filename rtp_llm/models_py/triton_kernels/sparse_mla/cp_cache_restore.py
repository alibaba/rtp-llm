import triton
import triton.language as tl


@triton.jit
def _restore_mla(
    Data,
    Map,
    Dest,
    Out,
    N,
    LOCAL,
    RANK_BYTES,
    OFFSET,
    SCATTER: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, 512)
    source = tl.load(Map + row, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (source >= 0)
    base = source // LOCAL * RANK_BYTES + source % LOCAL * 656 + OFFSET
    raw = tl.load(Data + base[:, None] + col[None, :], valid[:, None], other=0)
    sf = base[:, None] + 512 + col[None, :] // 128 * 4
    bits = tl.full((TILE, 512), 0, tl.uint32)
    for byte in tl.static_range(4):
        bits |= tl.load(Data + sf + byte, valid[:, None], other=0).to(
            tl.uint32
        ) << (byte * 8)
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32) * bits.to(
        tl.float32, bitcast=True
    )
    destination = tl.load(Dest + row, row < N, other=0) if SCATTER else row
    tl.store(
        Out + destination[:, None].to(tl.int64) * 576 + col[None, :],
        values.to(tl.bfloat16),
        valid[:, None],
    )
    rope = tl.arange(0, 64)
    lo = tl.load(
        Data + base[:, None] + 528 + rope[None, :] * 2, valid[:, None], other=0
    )
    hi = tl.load(
        Data + base[:, None] + 529 + rope[None, :] * 2, valid[:, None], other=0
    )
    bf = (lo.to(tl.uint16) | hi.to(tl.uint16) << 8).to(tl.bfloat16, bitcast=True)
    tl.store(
        Out + destination[:, None].to(tl.int64) * 576 + 512 + rope[None, :],
        bf,
        valid[:, None],
    )


@triton.jit
def _restore_indexer(
    Data,
    Map,
    K,
    S,
    N,
    LOCAL,
    WIDTH: tl.constexpr,
    HEAD: tl.constexpr,
    RANK_BYTES,
    OFFSET,
    COLS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)
    col = tl.arange(0, COLS)
    source = tl.load(Map + row, row < N, other=-1).to(tl.int64)
    valid = (row < N) & (source >= 0)
    base = source // LOCAL * RANK_BYTES + source % LOCAL * WIDTH + OFFSET
    value = tl.load(
        Data + base[:, None] + col[None, :],
        valid[:, None] & (col[None, :] < WIDTH),
        other=0,
    )
    tl.store(
        K + row[:, None].to(tl.int64) * HEAD + col[None, :],
        value,
        (row[:, None] < N) & (col[None, :] < HEAD),
    )
    tl.store(
        S + row[:, None].to(tl.int64) * (WIDTH - HEAD) + col[None, :] - HEAD,
        value,
        (row[:, None] < N) & (col[None, :] >= HEAD) & (col[None, :] < WIDTH),
    )
