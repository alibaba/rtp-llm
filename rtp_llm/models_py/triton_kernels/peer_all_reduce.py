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
