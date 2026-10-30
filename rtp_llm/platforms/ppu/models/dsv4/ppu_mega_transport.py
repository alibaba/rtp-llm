"""Pack routed tokens without converting BF16 activations or FP32 route weights."""
import triton
import triton.language as tl


@triton.jit
def pack_tokens(X, IDs, Weights, Active, Packet, COUNT: tl.constexpr,
                DIM: tl.constexpr, WORDS: tl.constexpr, HAS_MASK: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = row < COUNT
    if HAS_MASK:
        valid &= tl.load(Active + row, row < COUNT, False)
    base = Packet + row * WORDS
    px = tl.cast(base, tl.pointer_type(tl.bfloat16))
    x = tl.load(X + row * DIM + cols, (row < COUNT) & (cols < DIM), 0)
    tl.store(px + cols, x, cols < DIM)
    lanes = tl.arange(0, 8)
    ids = tl.load(IDs + row * 8 + lanes, valid, -1)
    weights = tl.load(Weights + row * 8 + lanes, valid, 0)
    pi = tl.cast(base + DIM // 2, tl.pointer_type(tl.int16))
    pw = tl.cast(base + DIM // 2 + 4, tl.pointer_type(tl.float32))
    tl.store(pi + lanes, ids.to(tl.int16))
    tl.store(pw + lanes, weights)


@triton.jit
def unpack_tokens(Packet, X, IDs, Weights, LocalIDs, LocalWeights,
                  DIM: tl.constexpr, WORDS: tl.constexpr,
                  BEGIN: tl.constexpr, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = Packet + row * WORDS
    cols = tl.arange(0, BLOCK)
    px = tl.cast(base, tl.pointer_type(tl.bfloat16))
    tl.store(X + row * DIM + cols, tl.load(px + cols, cols < DIM, 0), cols < DIM)
    lanes = tl.arange(0, 8)
    pi = tl.cast(base + DIM // 2, tl.pointer_type(tl.int16))
    pw = tl.cast(base + DIM // 2 + 4, tl.pointer_type(tl.float32))
    ids, weights = tl.load(pi + lanes).to(tl.int64), tl.load(pw + lanes)
    local = (ids >= BEGIN) & (ids < BEGIN + EXPERTS)
    tl.store(IDs + row * 8 + lanes, ids)
    tl.store(Weights + row * 8 + lanes, weights)
    tl.store(LocalIDs + row * 8 + lanes, tl.where(local, ids - BEGIN, -1))
    tl.store(LocalWeights + row * 8 + lanes, tl.where(local, weights, 0))


@triton.jit
def sum_contributions(Parts, Output, ELEMENTS: tl.constexpr, NODES: tl.constexpr,
                      BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    result = tl.full((BLOCK,), 0, tl.float32)
    for node in range(NODES):
        result += tl.load(Parts + node * ELEMENTS + offsets, offsets < ELEMENTS, 0).to(tl.float32)
    tl.store(Output + offsets, result, offsets < ELEMENTS)
