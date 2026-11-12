import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _prepare_draft_qsa(
    Q,
    POS,
    SOURCES,
    VISIBLE,
    FREQ,
    PADDED,
    COS,
    SIN,
    WEIGHTS,
    LENGTHS,
    QS: tl.constexpr,
    QH: tl.constexpr,
    H: tl.constexpr,
    INV_SCALE: tl.constexpr,
    FROM_METADATA: tl.constexpr = False,
    ROW_LENGTHS=None,
    PREFIX=None,
    CU=None,
    BATCH: tl.constexpr = 0,
    RB: tl.constexpr = 1,
    RAW_QUERY: tl.constexpr = False,
    QG=None,
    EPS: tl.constexpr = 1e-6,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    if FROM_METADATA:
        request, lane = row // 4, row % 4
        count = tl.load(ROW_LENGTHS + request)
        active = lane < count
        source = tl.where(active, tl.load(CU + request) + lane, 0)
        requests = tl.arange(0, RB)
        ends = tl.load(CU + requests + 1, requests < BATCH, other=2147483647)
        first_owner = tl.minimum(
            tl.sum(((0 >= ends) & (requests < BATCH)).to(tl.int32), 0), BATCH - 1
        )
        total = tl.load(CU + BATCH)
        first_position = tl.where(
            total > 0, tl.load(PREFIX + first_owner).to(tl.int64), 0
        )
        prefix = tl.load(PREFIX + request).to(tl.int64)
        position = tl.where(active, prefix + lane, first_position).to(tl.float32)
    else:
        source = tl.load(SOURCES + row)
        position = tl.load(POS + source).to(tl.float32)
    frequency = tl.load(FREQ + col % 32)
    angle = (position * INV_SCALE) * frequency
    cosine = libdevice.cos(angle).to(tl.bfloat16).to(tl.float32)
    sine = libdevice.sin(angle).to(tl.bfloat16).to(tl.float32)
    offset = source * QS + head * QH
    if RAW_QUERY:
        norm_lane = tl.arange(0, 32)
        norm_col = norm_lane * 4
        a = tl.load(Q + offset + norm_col).to(tl.float32)
        b = tl.load(Q + offset + norm_col + 1).to(tl.float32)
        c = tl.load(Q + offset + norm_col + 2).to(tl.float32)
        d = tl.load(Q + offset + norm_col + 3).to(tl.float32)
        variance = tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0
        inverse = tl.rsqrt(variance + EPS)
        raw = tl.load(Q + offset + col).to(tl.float32)
        gain = 1.0 + tl.load(QG + col).to(tl.float32)
        value = ((raw * inverse) * gain).to(tl.bfloat16)
        other = tl.gather(value, (col + 32) % 64, 0).to(tl.float32)
    else:
        value = tl.load(Q + offset + col)
        partner = tl.where(col < 32, col + 32, col - 32)
        other = tl.load(Q + offset + partner, col < 64, other=0).to(tl.float32)

    rotated = tl.where(col < 32, -other, other)
    first = (value.to(tl.float32) * cosine).to(tl.bfloat16).to(tl.float32)
    second = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = tl.where(col < 64, (first + second).to(tl.bfloat16), value)
    tl.store(PADDED + (row * H + head) * 128 + col, result)
    tl.store(WEIGHTS + row * H + head, 0.08838834764831845)
    if head == 0:
        # Writer RoPE remains in packed order, unlike the padded scorer query.
        if FROM_METADATA:
            owner = tl.minimum(
                tl.sum(((row >= ends) & (requests < BATCH)).to(tl.int32), 0), BATCH - 1
            )
            logical = tl.where(
                row < total,
                tl.load(PREFIX + owner).to(tl.int64)
                + row
                - tl.load(CU + owner).to(tl.int64),
                0,
            )
        else:
            logical = tl.load(POS + row).to(tl.int64)
        block_start = (logical - logical % 4).to(tl.float32)
        block_angle = (block_start * INV_SCALE) * frequency
        tl.store(COS + row * 64 + col, libdevice.cos(block_angle), col < 64)
        tl.store(SIN + row * 64 + col, libdevice.sin(block_angle), col < 64)
        if FROM_METADATA:
            visible = tl.where(active, prefix + lane + 1, 0).to(tl.int32)
            tl.store(VISIBLE + row, visible)
        else:
            visible = tl.load(VISIBLE + row)
        tl.store(LENGTHS + row, visible // 4)
