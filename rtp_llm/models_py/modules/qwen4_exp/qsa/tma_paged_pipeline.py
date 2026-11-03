"""TMA reads from HND paged KV with BF16 queries and FP32 cache scales."""

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
def _page(Table, kt, KN: gl.constexpr, PAGE_SIZE: gl.constexpr, WIDTH, PAGES):
    logical = kt * KN // PAGE_SIZE
    physical = gl.load(Table + logical, logical < WIDTH, other=0)
    valid = (physical > 0) & (physical < PAGES)
    return gl.where(valid, physical, 0), valid, (kt * KN) % PAGE_SIZE


@g.jit
def _producer(
    K,
    V,
    buf,
    full,
    empty,
    S,
    POSITIONS: gl.constexpr,
    G: gl.constexpr,
    KN: gl.constexpr,
    D: gl.constexpr,
    SLOTS: gl.constexpr,
    Tile,
    Table,
    Prefix,
    WIDTH,
    PAGES,
    Scales,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    RAW_FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    FRONTIER: gl.constexpr,
):
    tile = Tile
    tiles = gl.cdiv(gl.minimum(Prefix + S, Prefix + (tile + 1) * POSITIONS), KN)
    for operand in range(2 * tiles):
        slot = operand % SLOTS
        iteration = operand // SLOTS
        if operand >= SLOTS:
            mbarrier.wait(empty.index(slot), (iteration - 1) & 1)
        mbarrier.expect(full.index(slot), KN * D * (1 if RAW_FP8 else 2))
        kt = tiles - 1 - operand // 2 if FRONTIER else operand // 2
        physical, page_valid, page_offset = _page(
            Table, kt, KN, PAGE_SIZE, WIDTH, PAGES
        )
        if operand % 2 == 0:
            tma.async_copy_global_to_shared(
                K,
                [physical * CACHE_ROWS + page_offset, 0],
                full.index(slot),
                buf.index(slot),
            )
        else:
            tma.async_copy_global_to_shared(
                V,
                [physical * CACHE_ROWS + PAGE_SIZE + page_offset, 0],
                full.index(slot),
                buf.index(slot),
            )


@g.jit
def _attention_step(
    q0,
    q1,
    K,
    V,
    Mask,
    buf,
    full,
    empty,
    S,
    WORDS,
    D: gl.constexpr,
    G: gl.constexpr,
    QM: gl.constexpr,
    KN: gl.constexpr,
    PIPE: gl.constexpr,
    SLOTS: gl.constexpr,
    WM: gl.constexpr,
    EARLY: gl.constexpr,
    Table,
    Prefix,
    WIDTH,
    PAGES,
    Scales,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    RAW_FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    iteration,
    tiles,
    pos,
    live,
    c,
    maximum,
    normalizer,
    acc,
    CHECK_LENGTH: gl.constexpr,
    FRONTIER: gl.constexpr,
):
    mma: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, 4 // WM], instr_shape=[16, 8]
    )
    lhs: gl.constexpr = DotOperandLayout(0, mma, 2)
    rhs: gl.constexpr = DotOperandLayout(1, mma, 2)
    raw_rhs: gl.constexpr = DotOperandLayout(1, mma, 4 if RAW_FP8 else 2)
    kt = tiles - 1 - iteration if FRONTIER else iteration
    physical, page_valid, page_offset = _page(Table, kt, KN, PAGE_SIZE, WIDTH, PAGES)
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
        mbarrier.expect(full.index(ks), KN * D * (1 if RAW_FP8 else 2))
        tma.async_copy_global_to_shared(
            K,
            [physical * CACHE_ROWS + page_offset, 0],
            full.index(ks),
            buf.index(ks),
        )
        mbarrier.expect(full.index(vs), KN * D * (1 if RAW_FP8 else 2))
        tma.async_copy_global_to_shared(
            V,
            [physical * CACHE_ROWS + PAGE_SIZE + page_offset, 0],
            full.index(vs),
            buf.index(vs),
        )
    mbarrier.wait(full.index(ks), kp, deps=[buf])
    if RAW_FP8 and G == 6:
        k0 = buf.index(ks).slice(0, 128, dim=1).permute((1, 0)).load(raw_rhs)
        k0 = gl.convert_layout(k0.to(gl.bfloat16), rhs)
        scores = mma_v2(q0, k0, gl.full((QM, KN), 0.0, gl.float32, mma))
        k1 = buf.index(ks).slice(128, 128, dim=1).permute((1, 0)).load(raw_rhs)
        k1 = gl.convert_layout(k1.to(gl.bfloat16), rhs)
        scores = mma_v2(q1, k1, scores)
    else:
        k = buf.index(ks).permute((1, 0)).load(raw_rhs)
        if RAW_FP8:
            k = gl.convert_layout(k.to(gl.bfloat16), rhs)
        # FP8 values are exactly representable in BF16; keep Q unquantized.
        scores = mma_v2(q0, k, gl.full((QM, KN), 0.0, gl.float32, mma))
    column = kt * KN + c
    if SCALED:
        ks_value = gl.load(
            Scales + physical.to(gl.int64) * SCALE_PAGE + page_offset + c,
            page_valid & ((column < Prefix + S) if CHECK_LENGTH else True),
            other=0.0,
        )
        scores = scores * ks_value[None, :]
    scores = scores * (D**-0.5 * 1.4426950408889634)
    if PIPE:
        # All compute warps must finish their generic shared-memory reads before
        # the producer can overwrite this slot with another asynchronous transfer.
        gl.thread_barrier()
        fence_async_shared()
        mbarrier.arrive(empty.index(ks))
    column = kt * KN + c
    bits = gl.load(
        Mask + pos[:, None] * WORDS + column[None, :] // 32,
        live[:, None] & ((column[None, :] < Prefix + S) if CHECK_LENGTH else True),
        other=0,
    ).to(gl.uint32)
    selected = (((bits >> (column[None, :] % 32)) & 1) != 0) & page_valid
    scores = gl.where(selected, scores, -float("inf"))
    next_max = gl.maximum(maximum, gl.max(scores, 1))
    safe_max = gl.where(next_max == -float("inf"), 0.0, next_max)
    alpha = gl.exp2(maximum - safe_max)
    p = gl.exp2(scores - safe_max[:, None])
    # Retain the BF16 rounding residual while using warp MMA for both PV terms.
    if SCALED:
        vs_value = gl.load(
            Scales + physical.to(gl.int64) * SCALE_PAGE + SCALE_KV + page_offset + c,
            page_valid & ((column < Prefix + S) if CHECK_LENGTH else True),
            other=0.0,
        )
        # Fold per-key V scales into P while preserving its BF16 residual.
        weights = p * vs_value[None, :]
    else:
        weights = p
    hi = weights.to(gl.bfloat16)
    lo = (weights - hi.to(gl.float32)).to(gl.bfloat16)
    hi = gl.convert_layout(hi, lhs)
    lo = gl.convert_layout(lo, lhs)
    acc = acc * alpha[:, None]
    if EARLY:
        normalizer = normalizer * alpha + gl.sum(p, 1)
        maximum = next_max
    mbarrier.wait(full.index(vs), vp, deps=[buf])
    v = buf.index(vs).load(raw_rhs)
    if RAW_FP8:
        v = gl.convert_layout(v.to(gl.bfloat16), rhs)
    vr = gl.arange(0, KN, layout=SliceLayout(1, rhs))
    if CHECK_LENGTH:
        v = gl.where((page_valid & (kt * KN + vr < Prefix + S))[:, None], v, 0.0)
    else:
        v = gl.where(page_valid, v, 0.0)
    acc = mma_v2(hi, v, acc)
    acc = mma_v2(lo, v, acc)
    gl.thread_barrier()
    fence_async_shared()
    if PIPE:
        mbarrier.arrive(empty.index(vs))
    if not EARLY:
        normalizer = normalizer * alpha + gl.sum(p, 1)
        maximum = next_max
    return maximum, normalizer, acc


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
    Tile,
    Table,
    Prefix,
    WIDTH,
    PAGES,
    Scales,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    RAW_FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    FRONTIER: gl.constexpr,
):
    mma: gl.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[WM, 4 // WM], instr_shape=[16, 8]
    )
    lhs: gl.constexpr = DotOperandLayout(0, mma, 2)
    load_layout: gl.constexpr = BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    tile = Tile
    group = gl.program_id(0 if G == 4 else 1)
    rr = gl.arange(0, QM, layout=SliceLayout(1, load_layout))
    if RAW_FP8 and G == 6:
        dd = gl.arange(0, D // 2, layout=SliceLayout(0, load_layout))
        pp = tile * POSITIONS + rr // G
        hh = group * G + rr % G
        active = (rr < POSITIONS * G) & (pp < S)
        q0 = gl.load(
            Q + hh[:, None] * QH + pp[:, None] * QS + dd[None, :],
            active[:, None],
            other=0.0,
        )
        q0 = gl.convert_layout(q0, lhs)
        q1 = gl.load(
            Q + hh[:, None] * QH + pp[:, None] * QS + 128 + dd[None, :],
            active[:, None],
            other=0.0,
        )
        q1 = gl.convert_layout(q1, lhs)
    else:
        dd = gl.arange(0, D, layout=SliceLayout(0, load_layout))
        pp = tile * POSITIONS + rr // G
        hh = group * G + rr % G
        active = (rr < POSITIONS * G) & (pp < S)
        q = gl.load(
            Q + hh[:, None] * QH + pp[:, None] * QS + dd[None, :],
            active[:, None],
            other=0.0,
        )
        q0 = gl.convert_layout(q, lhs)
        q1 = q0
    r = gl.arange(0, QM, layout=SliceLayout(1, mma))
    c = gl.arange(0, KN, layout=SliceLayout(0, mma))
    pos = tile * POSITIONS + r // G
    live = (r < POSITIONS * G) & (pos < S)
    maximum = gl.full((QM,), -float("inf"), gl.float32, SliceLayout(1, mma))
    normalizer = gl.full((QM,), 0.0, gl.float32, SliceLayout(1, mma))
    acc = gl.full((QM, D), 0.0, gl.float32, mma)
    tiles = gl.cdiv(gl.minimum(Prefix + S, Prefix + (tile + 1) * POSITIONS), KN)
    # Only the first reverse tile can extend past the logical KV length.
    # Sparse membership and physical-page validity still apply to every tile.
    if FRONTIER:
        if tiles > 0:
            maximum, normalizer, acc = _attention_step(
                q0,
                q1,
                K,
                V,
                Mask,
                buf,
                full,
                empty,
                S,
                WORDS,
                D,
                G,
                QM,
                KN,
                PIPE,
                SLOTS,
                WM,
                EARLY,
                Table,
                Prefix,
                WIDTH,
                PAGES,
                Scales,
                PAGE_SIZE,
                SCALE_PAGE,
                SCALE_KV,
                RAW_FP8,
                SCALED,
                CACHE_ROWS,
                0,
                tiles,
                pos,
                live,
                c,
                maximum,
                normalizer,
                acc,
                True,
                FRONTIER,
            )
        for iteration in range(1, tiles):
            maximum, normalizer, acc = _attention_step(
                q0,
                q1,
                K,
                V,
                Mask,
                buf,
                full,
                empty,
                S,
                WORDS,
                D,
                G,
                QM,
                KN,
                PIPE,
                SLOTS,
                WM,
                EARLY,
                Table,
                Prefix,
                WIDTH,
                PAGES,
                Scales,
                PAGE_SIZE,
                SCALE_PAGE,
                SCALE_KV,
                RAW_FP8,
                SCALED,
                CACHE_ROWS,
                iteration,
                tiles,
                pos,
                live,
                c,
                maximum,
                normalizer,
                acc,
                False,
                FRONTIER,
            )
    else:
        for iteration in range(tiles):
            maximum, normalizer, acc = _attention_step(
                q0,
                q1,
                K,
                V,
                Mask,
                buf,
                full,
                empty,
                S,
                WORDS,
                D,
                G,
                QM,
                KN,
                PIPE,
                SLOTS,
                WM,
                EARLY,
                Table,
                Prefix,
                WIDTH,
                PAGES,
                Scales,
                PAGE_SIZE,
                SCALE_PAGE,
                SCALE_KV,
                RAW_FP8,
                SCALED,
                CACHE_ROWS,
                iteration,
                tiles,
                pos,
                live,
                c,
                maximum,
                normalizer,
                acc,
                True,
                FRONTIER,
            )
    output = gl.where(normalizer[:, None] > 0, acc / normalizer[:, None], 0.0)
    dc = gl.arange(0, D, layout=SliceLayout(0, mma))
    heads = group * G + r % G
    gl.store(
        O + heads[:, None] * OH + pos[:, None] * OS + dc[None, :],
        output.to(O.dtype.element_ty),
        live[:, None],
    )


@g.jit
def _execute_prefill(
    Q,
    K,
    V,
    Mask,
    O,
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
    WM: gl.constexpr,
    EARLY: gl.constexpr,
    Tile,
    Table,
    Prefix,
    WIDTH,
    PAGES,
    Scales,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    RAW_FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    FRONTIER: gl.constexpr,
):
    # K and V occupy successive slots in one ring, allowing V to arrive
    # while QK and softmax execute without allocating separate K/V pipelines.
    slots: gl.constexpr = 3 if PIPE else 2
    buf = gl.allocate_shared_memory(K.dtype, [slots, KN, D], K.layout)
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
                        Tile,
                        Table,
                        Prefix,
                        WIDTH,
                        PAGES,
                        Scales,
                        PAGE_SIZE,
                        SCALE_PAGE,
                        SCALE_KV,
                        RAW_FP8,
                        SCALED,
                        CACHE_ROWS,
                        FRONTIER,
                    ),
                ),
                (
                    _producer,
                    (
                        K,
                        V,
                        buf,
                        full,
                        empty,
                        S,
                        POSITIONS,
                        G,
                        KN,
                        D,
                        slots,
                        Tile,
                        Table,
                        Prefix,
                        WIDTH,
                        PAGES,
                        Scales,
                        PAGE_SIZE,
                        SCALE_PAGE,
                        SCALE_KV,
                        RAW_FP8,
                        SCALED,
                        CACHE_ROWS,
                        FRONTIER,
                    ),
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
            Tile,
            Table,
            Prefix,
            WIDTH,
            PAGES,
            Scales,
            PAGE_SIZE,
            SCALE_PAGE,
            SCALE_KV,
            RAW_FP8,
            SCALED,
            CACHE_ROWS,
            FRONTIER,
        )
    for i in gl.static_range(slots):
        mbarrier.invalidate(full.index(i))
        mbarrier.invalidate(empty.index(i))


@g.jit(do_not_specialize=["S", "WORDS", "Prefix", "WIDTH", "PAGES"])
def _paged_prefill(
    Q,
    Cache,
    Table,
    Scales,
    Mask,
    O,
    S,
    WORDS,
    Prefix,
    WIDTH,
    PAGES,
    PAGE_SIZE: gl.constexpr,
    SCALE_PAGE: gl.constexpr,
    SCALE_KV: gl.constexpr,
    RAW_FP8: gl.constexpr,
    SCALED: gl.constexpr,
    CACHE_ROWS: gl.constexpr,
    G: gl.constexpr,
    QM: gl.constexpr,
    QH: gl.constexpr,
    QS: gl.constexpr,
    OS: gl.constexpr,
    PIPE: gl.constexpr,
    EARLY: gl.constexpr,
    FRONTIER: gl.constexpr,
    KN: gl.constexpr,
):
    positions: gl.constexpr = QM // G
    tile = gl.cdiv(S, positions) - 1 - gl.program_id(1 if G == 4 else 0)
    _execute_prefill(
        Q,
        Cache,
        Cache,
        Mask,
        O,
        S,
        WORDS,
        256,
        G,
        QM,
        KN,
        positions,
        QH,
        QS,
        256,
        OS,
        PIPE,
        QM // 16,
        EARLY,
        tile,
        Table,
        Prefix,
        WIDTH,
        PAGES,
        Scales,
        PAGE_SIZE,
        SCALE_PAGE,
        SCALE_KV,
        RAW_FP8,
        SCALED,
        CACHE_ROWS,
        FRONTIER,
    )


def run_prefill(q, cache, table, selected, prefix, page_size, scale):
    from .tma_prefill import _build_membership

    _, heads, tokens, dim = q.shape
    scaled = cache.dtype == torch.float8_e4m3fn
    qm, group = (
        (16, 12) if heads == 12 and tokens <= 512 else (32, 4 if heads == 12 else 6)
    )
    words = triton.cdiv(prefix + tokens, 32)
    membership = torch.empty((tokens, words), device=q.device, dtype=torch.int32)
    if scaled and prefix + tokens >= 1024:
        from .prefill_staging import stage_paged_kv

        cache, table, scale = stage_paged_kv(
            cache,
            table,
            scale,
            prefix + tokens,
            page_size,
            selected=selected,
            membership=membership,
        )
    else:
        _build_membership[(tokens,)](
            selected,
            membership,
            prefix + tokens,
            words,
            16 if prefix + tokens <= 512 else 128,
            num_warps=4,
        )
    pages = cache.shape[0]
    fp8 = cache.dtype == torch.float8_e4m3fn
    key_tile = 32
    layout = NVMMASharedLayout(128, 8 if fp8 else 16)
    cache_rows = cache.stride(0) // dim
    # Flatten the row-padded HND pool into a 2D descriptor. Page lookup skips
    # the padding without copying or declaring a higher-dimensional TMA tile.
    descriptor = TensorDescriptor(
        cache, [pages * cache_rows, dim], [dim, 1], [key_tile, dim], layout
    )
    output = torch.empty(
        (1, tokens, heads, dim), dtype=q.dtype, device=q.device
    ).transpose(1, 2)
    grid = (triton.cdiv(tokens, qm // group), heads // group)
    if group == 4:
        grid = grid[::-1]
    pipe = (not scaled or heads == 6) and prefix + tokens <= 2048
    _paged_prefill[grid](
        q,
        descriptor,
        table,
        cache if scale is None else scale,
        membership,
        output,
        tokens,
        words,
        prefix,
        table.shape[1],
        pages,
        page_size,
        0 if scale is None else scale.stride(0),
        0 if scale is None else scale.stride(1),
        fp8,
        scaled,
        cache_rows,
        group,
        qm,
        q.stride(1),
        q.stride(2),
        output.stride(2),
        pipe,
        True,
        scaled or heads == 12 or not pipe,
        key_tile,
        num_warps=4,
    )
    return output
