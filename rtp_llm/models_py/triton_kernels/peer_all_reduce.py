# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _canonical_peer_sum(
    P0, P1, P2, P3, P4, P5, P6, P7, Y, N, BLOCK: tl.constexpr, ALIGNED: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Full tiles retain vector loads even though N stays a runtime scalar.
    mask = True if ALIGNED else index < N
    value = tl.load(P0 + index, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P1 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P2 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P3 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P4 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P5 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P6 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    value = value + tl.load(P7 + index, mask, other=0, cache_modifier=".cg").to(
        tl.float32
    )
    tl.store(Y + index, value, mask)


@triton.jit(do_not_specialize=["N"])
def _stage_peer_pingpong(
    X,
    BUFFER,
    SLOT,
    N,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOT) ^ 1
    mask = True if ALIGNED else i < N
    value = tl.load(X + i, mask, other=0)
    tl.store(BUFFER + slot * CAPACITY + i, value, mask)


@triton.jit(do_not_specialize=["N"])
def _canonical_pingpong_sum(
    P0,
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
    P7,
    Y,
    SLOT,
    N,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOT)
    j = slot * CAPACITY + i
    mask = True if ALIGNED else i < N
    value = tl.load(P0 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P1 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P2 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P3 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P4 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P5 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P6 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    value = value + tl.load(P7 + j, mask, other=0, cache_modifier=".cg").to(tl.float32)
    tl.store(Y + i, value, mask)


@triton.jit
def _advance_pingpong_peer_barrier(
    S0, S1, S2, S3, S4, S5, S6, S7, SLOT, RANK: tl.constexpr
):
    # A phase uses its own channel. Self signaling is harmless and keeps the
    # eight-lane CAS loops uniform; completed lanes do not consume new signals.
    lane = tl.arange(0, 8)
    channel = tl.load(SLOT) ^ 1
    # Staging completed on this stream. Advance once here, before peer reads.
    tl.store(SLOT, channel)
    target = tl.where(
        lane == 0,
        S0,
        tl.where(
            lane == 1,
            S1,
            tl.where(
                lane == 2,
                S2,
                tl.where(
                    lane == 3,
                    S3,
                    tl.where(
                        lane == 4,
                        S4,
                        tl.where(lane == 5, S5, tl.where(lane == 6, S6, S7)),
                    ),
                ),
            ),
        ),
    )
    own = tl.where(
        RANK == 0,
        S0,
        tl.where(
            RANK == 1,
            S1,
            tl.where(
                RANK == 2,
                S2,
                tl.where(
                    RANK == 3,
                    S3,
                    tl.where(
                        RANK == 4,
                        S4,
                        tl.where(RANK == 5, S5, tl.where(RANK == 6, S6, S7)),
                    ),
                ),
            ),
        ),
    )
    target = target.to(tl.pointer_type(tl.uint32))
    own = own.to(tl.pointer_type(tl.uint32))
    zero = tl.full((8,), 0, tl.uint32)
    one = tl.full((8,), 1, tl.uint32)
    now = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    deadline = now + 10000000000
    done = tl.full((8,), False, tl.int1)
    while tl.sum((~done).to(tl.int32), 0) > 0:
        old = tl.atomic_cas(
            target + channel * 8 + RANK,
            tl.where(done, one, zero),
            one,
            sem="release",
            scope="sys",
        )
        done = done | (old == 0)
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            constraints="=l",
            args=[],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
        if now > deadline:
            tl.inline_asm_elementwise(
                "trap;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
    done = tl.full((8,), False, tl.int1)
    while tl.sum((~done).to(tl.int32), 0) > 0:
        old = tl.atomic_cas(
            own + channel * 8 + lane,
            tl.where(done, zero, one),
            zero,
            sem="acquire",
            scope="sys",
        )
        done = done | (old == 1)
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            constraints="=l",
            args=[],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
        if now > deadline:
            tl.inline_asm_elementwise(
                "trap;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
