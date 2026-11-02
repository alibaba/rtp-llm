import triton
import triton.language as tl


@triton.jit
def pack_sparse(X, IDs, Weights, Active, Packet, Inverse, COUNT: tl.constexpr,
                CAP: tl.constexpr, DIM: tl.constexpr, WORDS: tl.constexpr,
                STRIDE: tl.constexpr, EXPERTS: tl.constexpr, HAS_MASK: tl.constexpr,
                ROWS: tl.constexpr, BLOCK: tl.constexpr):
    peer, row = tl.program_id(0), tl.program_id(1)
    rows, lanes = tl.arange(0, ROWS), tl.arange(0, 8)
    ids = tl.load(IDs + rows[:, None] * 8 + lanes[None, :], rows[:, None] < COUNT, -1)
    active = rows < COUNT
    if HAS_MASK:
        active &= tl.load(Active + rows, rows < COUNT, False)
    hit = active & (tl.sum(((ids >= peer * EXPERTS) & (ids < (peer + 1) * EXPERTS)).to(tl.int32), 1) > 0)
    total = tl.sum(hit.to(tl.int32), 0)
    offset = tl.sum((hit & (rows < row)).to(tl.int32), 0)
    valid = tl.sum((hit & (rows == row)).to(tl.int32), 0) > 0
    tl.store(Inverse + peer * CAP + row, tl.where(valid, offset, -1))
    if row == 0:
        tl.store(Packet + peer * STRIDE, total)
    base = Packet + peer * STRIDE + 4 + offset * WORDS
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * DIM + cols, (row < COUNT) & (cols < DIM), 0)
    tl.store(tl.cast(base, tl.pointer_type(tl.bfloat16)) + cols, x, valid & (cols < DIM))
    enabled = row < COUNT
    if HAS_MASK:
        enabled &= tl.load(Active + row, row < COUNT, False)
    own_ids = tl.load(IDs + row * 8 + lanes, enabled, -1)
    own_weights = tl.load(Weights + row * 8 + lanes, enabled, 0)
    tl.store(tl.cast(base + DIM // 2, tl.pointer_type(tl.int16)) + lanes, own_ids.to(tl.int16), valid)
    tl.store(tl.cast(base + DIM // 2 + 4, tl.pointer_type(tl.float32)) + lanes, own_weights, valid)


@triton.jit
def unpack_sparse(Packet, X, IDs, Weights, CAP: tl.constexpr, DIM: tl.constexpr,
                  WORDS: tl.constexpr, STRIDE: tl.constexpr, NODE: tl.constexpr,
                  EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    source, row = tl.program_id(0), tl.program_id(1)
    count = tl.load(Packet + source * STRIDE)
    valid = row < count
    base = Packet + source * STRIDE + 4 + row * WORDS
    out = source * CAP + row
    cols, lanes = tl.arange(0, BLOCK), tl.arange(0, 8)
    px = tl.cast(base, tl.pointer_type(tl.bfloat16))
    tl.store(X + out * DIM + cols, tl.load(px + cols, valid & (cols < DIM), 0), cols < DIM)
    ids = tl.load(tl.cast(base + DIM // 2, tl.pointer_type(tl.int16)) + lanes, valid, -1).to(tl.int64)
    weights = tl.load(tl.cast(base + DIM // 2 + 4, tl.pointer_type(tl.float32)) + lanes, valid, 0)
    owned = (ids >= NODE * EXPERTS) & (ids < (NODE + 1) * EXPERTS)
    tl.store(IDs + out * 8 + lanes, tl.where(owned, ids - NODE * EXPERTS, -1))
    tl.store(Weights + out * 8 + lanes, tl.where(owned, weights, 0))


@triton.jit
def pack_return(Y, DispatchPacket, Packet, CAP: tl.constexpr,
                DIM: tl.constexpr, DISPATCH_STRIDE: tl.constexpr,
                STRIDE: tl.constexpr,
                BLOCK: tl.constexpr):
    peer, row = tl.program_id(0), tl.program_id(1)
    count = tl.load(DispatchPacket + peer * DISPATCH_STRIDE)
    if row == 0:
        tl.store(Packet + peer * STRIDE, count)
    cols = tl.arange(0, BLOCK)
    value = tl.load(Y + (peer * CAP + row) * DIM + cols, (row < count) & (cols < DIM), 0)
    dest = tl.cast(Packet + peer * STRIDE + 4, tl.pointer_type(tl.bfloat16))
    tl.store(dest + row * DIM + cols, value, (row < count) & (cols < DIM))


@triton.jit
def unpack_return(Packet, Inverse, Output, CAP: tl.constexpr, DIM: tl.constexpr,
                  STRIDE: tl.constexpr, NODES: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    for peer in range(NODES):
        index = tl.load(Inverse + peer * CAP + row)
        src = tl.cast(Packet + peer * STRIDE + 4, tl.pointer_type(tl.bfloat16))
        result += tl.load(src + index * DIM + cols, (index >= 0) & (cols < DIM), 0).to(tl.float32)
    tl.store(Output + row * DIM + cols, result, cols < DIM)
