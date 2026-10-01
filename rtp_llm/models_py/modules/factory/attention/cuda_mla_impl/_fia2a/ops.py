"""Query packing and local/peer LSE reductions."""

import triton
import triton.language as tl


@triton.jit
def _a2a_pull(Peers, Received, Words: tl.constexpr, Rank: tl.constexpr,
              World: tl.constexpr, BLOCK: tl.constexpr):
    program = tl.program_id(0).to(tl.int64)
    # Interleave peers in the CTA order; keep all chunks on grid.x so large
    # payloads do not encounter grid.y's 65535 limit.
    source = (program + Rank) % World
    offset = (program // World) * BLOCK + tl.arange(0, BLOCK)
    words = tl.full((), Words, tl.int64)
    # Attach alignment after int_to_ptr: an integer hint is lost by Triton 3.6.
    peer = tl.multiple_of(tl.load(Peers + source).to(tl.pointer_type(tl.int32)), 16)
    value = tl.load(peer + tl.full((), Rank, tl.int64) * words + offset,
                    offset < words, other=0, cache_modifier=".cg")
    # Integer copying preserves the FP32 LSE bits in the two final BF16 slots.
    tl.store(Received + source * words + offset, value, offset < words)


@triton.jit
def _pack(Src, Dst, T: tl.constexpr, H: tl.constexpr,
          WORDS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, feature = index // WORDS, index % WORDS
    token, head = row // H, row % H
    source = (head * T + token) * WORDS + feature
    value = tl.load(Src + source, index < T * H * WORDS, other=0)
    tl.store(Dst + index, value, index < T * H * WORDS)


@triton.jit
def merge_local_splits(
    Partial, LSE, Output, OutputLSE,
    ROWS: tl.constexpr, HEADS: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr,
):
    part = tl.arange(0, BLOCK_S)
    if BLOCK_R == 1:
        # Preserve the original 2-D S×D reduction and CTA order for large S.
        row = tl.program_id(0).to(tl.int64) * HEADS + tl.program_id(1)
        band = tl.program_id(2)
        row_valid = tl.full((), True, tl.int1)
        offset = part.to(tl.int64) * (ROWS * HEADS) + row
        record_valid = part < SPLITS
    else:
        row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        band = tl.program_id(1)
        row_valid = row < ROWS * HEADS
        offset = part[None, :].to(tl.int64) * (ROWS * HEADS) + row[:, None]
        record_valid = row_valid[:, None] & (part[None, :] < SPLITS)
    dim = band * BLOCK_D + tl.arange(0, BLOCK_D)
    logs = tl.load(LSE + offset, record_valid, other=-float("inf"))
    valid = record_valid & (logs != -float("inf"))
    nonempty = tl.sum(valid.to(tl.int32), -1) > 0
    maximum = tl.where(nonempty, tl.max(logs, -1), 0.)
    weights = tl.where(valid, tl.exp2(logs - tl.expand_dims(maximum, -1)), 0.)
    # Empty splits leave O unwritten. Do not read their stale or NaN storage.
    values = tl.load(Partial + tl.expand_dims(offset, -1) * 512 + dim,
                     tl.expand_dims(valid, -1) & (dim < 512), other=0).to(tl.float32)
    denominator = tl.sum(weights, -1)
    result = tl.sum(values * tl.expand_dims(weights, -1), -2) / tl.expand_dims(tl.where(nonempty, denominator, 1.), -1)
    has_nan = tl.sum((logs != logs).to(tl.int32), -1) > 0
    result = tl.where(tl.expand_dims(has_nan, -1), float("nan"), result)
    tl.store(Output + tl.expand_dims(row, -1) * 512 + dim, result, tl.expand_dims(row_valid, -1) & (dim < 512))
    if band == 0:
        merged_lse = tl.where(nonempty, maximum + tl.log2(denominator), -float("inf"))
        tl.store(OutputLSE + row, tl.where(has_nan, float("nan"), merged_lse), row_valid)


@triton.jit
def _merge_splits_serial(W, L, O, ROWS: tl.constexpr, HEADS: tl.constexpr,
                         CAPACITY: tl.constexpr, WORLD: tl.constexpr, SPLITS: tl.constexpr,
                         BAND: tl.constexpr, OUT_ROW: tl.constexpr, OUT_HEAD: tl.constexpr,
                         BLOCK_D: tl.constexpr):
    dim = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    # Source-major wire storage can exceed 2^31 elements at large B*Q.
    # Put rows on grid.x: grid.y is limited to 65535, below legal B*Q.
    row, head = tl.program_id(0).to(tl.int64), tl.program_id(2).to(tl.int64)
    maximum = tl.full((), -float('inf'), tl.float32)
    for state in tl.static_range(WORLD * SPLITS):
        sender, part = state // SPLITS, state % SPLITS
        lse = tl.load(L + (sender * CAPACITY + part * ROWS + row) * HEADS + head)
        maximum = tl.maximum(maximum, lse, propagate_nan=tl.PropagateNan.ALL)
    any_valid = maximum != -float('inf')
    maximum = tl.where(any_valid, maximum, 0.)
    numerator = tl.full((BLOCK_D,), 0., tl.float32)
    denominator = tl.full((), 0., tl.float32)
    for state in tl.static_range(WORLD * SPLITS):
        sender, part = state // SPLITS, state % SPLITS
        unit = part * ROWS + row
        lse = tl.load(L + (sender * CAPACITY + unit) * HEADS + head)
        valid = lse != -float('inf')
        weight = tl.where(valid, tl.exp2(lse - maximum), 0.)
        offset = (((sender * CAPACITY + unit) * (512 // BAND) + dim // BAND)
                  * HEADS + head) * BAND + dim % BAND
        value = tl.load(W + offset, mask=valid & (dim < 512), other=0).to(tl.float32)
        numerator += value * weight
        denominator += weight
    result = numerator / tl.where(any_valid, denominator, 1.)
    tl.store(O + row * OUT_ROW + head * OUT_HEAD + dim, result, mask=dim < 512)
