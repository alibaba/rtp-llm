"""Explicit TMA and warp-MMA scheduling for SM120 causal QSA prefill.

Imported only after the shape, device and selection-provenance checks.
"""

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    NVMMADistributedLayout,
    NVMMASharedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    mbarrier,
    mma_v2,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@g.jit
def _producer(
    K,
    V,
    buf,
    full,
    empty,
    S,
    POSITIONS: gl.constexpr,
    KN: gl.constexpr,
    D: gl.constexpr,
    SLOTS: gl.constexpr,
):
    tile = gl.cdiv(S, POSITIONS) - 1 - gl.program_id(0)
    tiles = gl.cdiv(gl.minimum(S, (tile + 1) * POSITIONS), KN)
    for operand in range(2 * tiles):
        slot = operand % SLOTS
        iteration = operand // SLOTS
        if operand >= SLOTS:
            mbarrier.wait(empty.index(slot), (iteration - 1) & 1)
        mbarrier.expect(full.index(slot), KN * D * 2)
        kt = operand // 2
        if operand % 2 == 0:
            tma.async_copy_global_to_shared(
                K, [kt * KN, 0], full.index(slot), buf.index(slot)
            )
        else:
            tma.async_copy_global_to_shared(
                V, [kt * KN, 0], full.index(slot), buf.index(slot)
            )


@g.jit
def _consumer(
    Q,
    K,
    V,
    Mask,
    O,
    buf,
    full,
    empty,
    S,
    WORDS,
    D: gl.constexpr,
    G: gl.constexpr,
    QM: gl.constexpr,
    KN: gl.constexpr,
    POSITIONS: gl.constexpr,
    QH: gl.constexpr,
    QS: gl.constexpr,
    OH: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
    SLOTS: gl.constexpr,
    WM: gl.constexpr,
    EARLY: gl.constexpr,
):
    mma: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, 4 // WM], instr_shape=[16, 8]
    )
    lhs: gl.constexpr = DotOperandLayout(0, mma, 2)
    rhs: gl.constexpr = DotOperandLayout(1, mma, 2)
    load_layout: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    tile = gl.cdiv(S, POSITIONS) - 1 - gl.program_id(0)
    group = gl.program_id(1)
    rr = gl.arange(0, QM, layout=SliceLayout(1, load_layout))
    dd = gl.arange(0, D, layout=SliceLayout(0, load_layout))
    pp = tile * POSITIONS + rr // G
    hh = group * G + rr % G
    active = (rr < POSITIONS * G) & (pp < S)
    q = gl.load(
        Q + hh[:, None] * QH + pp[:, None] * QS + dd[None, :],
        active[:, None],
        other=0.0,
    )
    q = gl.convert_layout(q, lhs)
    r = gl.arange(0, QM, layout=SliceLayout(1, mma))
    c = gl.arange(0, KN, layout=SliceLayout(0, mma))
    pos = tile * POSITIONS + r // G
    live = (r < POSITIONS * G) & (pos < S)
    maximum = gl.full((QM,), -float("inf"), gl.float32, SliceLayout(1, mma))
    normalizer = gl.full((QM,), 0.0, gl.float32, SliceLayout(1, mma))
    acc = gl.full((QM, D), 0.0, gl.float32, mma)
    tiles = gl.cdiv(gl.minimum(S, (tile + 1) * POSITIONS), KN)
    for iteration in range(tiles):
        kt = iteration
        if PIPE:
            ks = (2 * iteration) % SLOTS
            vs = (2 * iteration + 1) % SLOTS
            kp = ((2 * iteration) // SLOTS) & 1
            vp = ((2 * iteration + 1) // SLOTS) & 1
        else:
            ks = 0
            vs = 1
            kp = iteration & 1
            vp = iteration & 1
            mbarrier.expect(full.index(ks), KN * D * 2)
            tma.async_copy_global_to_shared(
                K, [kt * KN, 0], full.index(ks), buf.index(ks)
            )
            mbarrier.expect(full.index(vs), KN * D * 2)
            tma.async_copy_global_to_shared(
                V, [kt * KN, 0], full.index(vs), buf.index(vs)
            )
        mbarrier.wait(full.index(ks), kp, deps=[buf])
        k = buf.index(ks).permute((1, 0)).load(rhs)
        scores = mma_v2(q, k, gl.full((QM, KN), 0.0, gl.float32, mma)) * (
            D**-0.5 * 1.4426950408889634
        )
        if PIPE:
            # All compute warps must finish their generic shared-memory reads before
            # the producer can overwrite this slot with another asynchronous transfer.
            gl.thread_barrier()
            fence_async_shared()
            mbarrier.arrive(empty.index(ks))
        column = kt * KN + c
        bits = gl.load(
            Mask + pos[:, None] * WORDS + column[None, :] // 32,
            live[:, None] & (column[None, :] < S),
            other=0,
        ).to(gl.uint32)
        selected = ((bits >> (column[None, :] % 32)) & 1) != 0
        scores = gl.where(selected, scores, -float("inf"))
        next_max = gl.maximum(maximum, gl.max(scores, 1))
        safe_max = gl.where(next_max == -float("inf"), 0.0, next_max)
        alpha = gl.exp2(maximum - safe_max)
        p = gl.exp2(scores - safe_max[:, None])
        # Retain the BF16 rounding residual while using warp MMA for both PV terms.
        hi = p.to(gl.bfloat16)
        lo = (p - hi.to(gl.float32)).to(gl.bfloat16)
        hi = gl.convert_layout(hi, lhs)
        lo = gl.convert_layout(lo, lhs)
        acc = acc * alpha[:, None]
        if EARLY:
            normalizer = normalizer * alpha + gl.sum(p, 1)
            maximum = next_max
        mbarrier.wait(full.index(vs), vp, deps=[buf])
        v = buf.index(vs).load(rhs)
        acc = mma_v2(hi, v, acc)
        acc = mma_v2(lo, v, acc)
        gl.thread_barrier()
        fence_async_shared()
        if PIPE:
            mbarrier.arrive(empty.index(vs))
        if not EARLY:
            normalizer = normalizer * alpha + gl.sum(p, 1)
            maximum = next_max
    output = gl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
    dc = gl.arange(0, D, layout=SliceLayout(0, mma))
    heads = group * G + r % G
    gl.store(
        O + heads[:, None] * OH + pos[:, None] * OS + dc[None, :],
        output.to(O.dtype.element_ty),
        live[:, None],
    )


@g.jit(do_not_specialize=["S", "WORDS"])
def _pipelined_prefill(
    Q,
    K,
    V,
    Mask,
    O,
    S,
    WORDS,
    G: gl.constexpr,
    QM: gl.constexpr,
    QH: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
    EARLY: gl.constexpr,
):
    D: gl.constexpr = 256
    KN: gl.constexpr = 32
    POSITIONS: gl.constexpr = QM // G
    WM: gl.constexpr = QM // 16
    OH: gl.constexpr = D
    # K and V occupy successive slots in one ring, allowing V to arrive
    # while QK and softmax execute without allocating separate K/V pipelines.
    slots: gl.constexpr = 3 if PIPE else 2
    buf = gl.allocate_shared_memory(gl.bfloat16, [slots, KN, D], K.layout)
    full = gl.allocate_shared_memory(gl.int64, [slots, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [slots, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(slots):
        mbarrier.init(full.index(i), count=1)
        mbarrier.init(empty.index(i), count=1)
    fence_async_shared()
    if PIPE:
        gl.warp_specialize(
            [
                (
                    _consumer,
                    (
                        Q,
                        K,
                        V,
                        Mask,
                        O,
                        buf,
                        full,
                        empty,
                        S,
                        WORDS,
                        D,
                        G,
                        QM,
                        KN,
                        POSITIONS,
                        QH,
                        QS,
                        OH,
                        OS,
                        True,
                        slots,
                        WM,
                        EARLY,
                    ),
                ),
                (
                    _producer,
                    (K, V, buf, full, empty, S, POSITIONS, KN, D, slots),
                ),
            ],
            worker_num_warps=[4],
            worker_num_regs=[24],
        )
    else:
        _consumer(
            Q,
            K,
            V,
            Mask,
            O,
            buf,
            full,
            empty,
            S,
            WORDS,
            D,
            G,
            QM,
            KN,
            POSITIONS,
            QH,
            QS,
            OH,
            OS,
            False,
            slots,
            WM,
            EARLY,
        )
    for i in gl.static_range(slots):
        mbarrier.invalidate(full.index(i))
        mbarrier.invalidate(empty.index(i))


_KV_LAYOUT = NVMMASharedLayout(128, 16)


def run_prefill(q, k, v, membership):
    _, heads, tokens, dim = q.shape
    query_tile, group_heads = (
        (16, 12) if heads == 12 and tokens <= 512 else (32, 4 if heads == 12 else 6)
    )
    # A separate producer pays off when enough short queries run in parallel.
    # Longer query tiles need the occupancy available with two operand slots.
    pipeline = heads == 6 and tokens <= 1024
    early_state = query_tile == 32
    key_tile = 32
    words = triton.cdiv(tokens, 32)
    output = torch.empty(
        (1, tokens, heads, dim), dtype=q.dtype, device=q.device
    ).transpose(1, 2)
    layout = _KV_LAYOUT
    key = TensorDescriptor(k, [tokens, dim], [k.stride(2), 1], [key_tile, dim], layout)
    value = TensorDescriptor(
        v, [tokens, dim], [v.stride(2), 1], [key_tile, dim], layout
    )
    positions = query_tile // group_heads
    _pipelined_prefill[(triton.cdiv(tokens, positions), heads // group_heads)](
        q,
        key,
        value,
        membership,
        output,
        tokens,
        words,
        group_heads,
        query_tile,
        q.stride(1),
        q.stride(2),
        output.stride(2),
        pipeline,
        early_state,
        num_warps=4,
    )
    return output
