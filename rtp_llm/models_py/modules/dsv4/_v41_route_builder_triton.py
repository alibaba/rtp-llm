# SPDX-License-Identifier: Apache-2.0
"""Row-owned copies for contiguous FP8 expert dispatch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _route_builder_kernel(
    SOURCE,  # uint8 [N, D] activation payload bytes
    SCALES,  # uint8 [N, S] linear UE8M0 scale bytes
    WEIGHTS,  # uint8 [N*K, 4] fp32 route-weight words viewed as bytes
    ORDER,  # int64 [A] active-order route ids (truncated prefix)
    COUNTS,  # int64 [E] per-expert actual (unpadded) counts
    STARTS,  # int64 [E] per-expert start in the active order
    BASE,  # int64 [E] per-expert padded segment starts
    ROUTED_X,  # uint8 [P, D]
    ROUTED_S,  # uint8 [P, S]
    ROUTED_W,  # uint8 [P, 4]
    VALID,  # uint8 [P] (bool view)
    ROUTE_MAP,  # int64 [N*K]
    P,
    A,
    E,
    TOPK: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    DPOW2: tl.constexpr,
    SPOW2: tl.constexpr,
    EPOW2: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    p0 = tl.program_id(0).to(tl.int64) * BLOCK_P
    pvec = p0 + tl.arange(0, BLOCK_P)  # [BLOCK_P] output rows
    pmask = pvec < P
    # owner lookup: load the segment starts ONCE per program, then count
    # starts <= p per row (the count-based upper bound; amortized over the tile).
    es = tl.arange(0, EPOW2)
    base_v = tl.load(BASE + es, mask=es < E, other=2147483647).to(tl.int64)  # [EPOW2]
    le = (base_v[None, :] <= pvec[:, None]) & (es[None, :] < E)
    e = tl.sum(le.to(tl.int32), axis=1) - 1  # [BLOCK_P]
    e = tl.maximum(e, 0).to(tl.int64)
    base_e = tl.load(BASE + e, mask=pmask, other=0)
    cnt_e = tl.load(COUNTS + e, mask=pmask, other=0)
    start_e = tl.load(STARTS + e, mask=pmask, other=0)
    j = pvec - base_e
    live = pmask & (j >= 0) & (j < cnt_e)
    k = start_e + j
    route = tl.load(ORDER + k, mask=live & (k >= 0) & (k < A), other=-1).to(tl.int64)
    src_row = route // TOPK

    dcols = tl.arange(0, DPOW2)
    dm = dcols < D
    xv = tl.load(
        SOURCE + src_row[:, None] * D + dcols[None, :],
        mask=live[:, None] & dm[None, :],
        other=0,
    )
    tl.store(
        ROUTED_X + pvec[:, None] * D + dcols[None, :],
        xv,
        mask=pmask[:, None] & dm[None, :],
    )

    scols = tl.arange(0, SPOW2)
    sm = scols < S
    sv = tl.load(
        SCALES + src_row[:, None] * S + scols[None, :],
        mask=live[:, None] & sm[None, :],
        other=0,
    )
    tl.store(
        ROUTED_S + pvec[:, None] * S + scols[None, :],
        sv,
        mask=pmask[:, None] & sm[None, :],
    )

    wcols = tl.arange(0, 4)
    wv = tl.load(
        WEIGHTS + route[:, None] * 4 + wcols[None, :],
        mask=live[:, None] & (wcols >= 0)[None, :],
        other=0,
    )
    tl.store(ROUTED_W + pvec[:, None] * 4 + wcols[None, :], wv, mask=pmask[:, None])

    tl.store(VALID + pvec, live.to(tl.int8), mask=pmask)
    tl.store(ROUTE_MAP + route, pvec, mask=live)


def route_builder(
    source,
    input_scales,
    weights,
    order,
    counts,
    starts,
    base,
    allocated,
    active_count,
    experts,
    topk,
    route_map,
):
    """Copy live routes into padded expert rows and fill their inverse map."""
    _, D = source.shape
    S = input_scales.shape[1]
    P = int(allocated)
    A = int(active_count)
    E = int(experts)
    device = source.device
    routed_x = torch.empty((P, D), dtype=torch.uint8, device=device)
    routed_s = torch.empty((P, S), dtype=torch.uint8, device=device)
    routed_w = torch.empty((P,), dtype=torch.float32, device=device)
    valid = torch.empty((P,), dtype=torch.bool, device=device)
    if P:
        grid = ((P + 3) // 4,)  # BLOCK_P = 4 rows per program
        _route_builder_kernel[grid](
            source,
            input_scales,
            weights.flatten().view(torch.uint8).reshape(-1, 4),
            order,
            counts,
            starts,
            base,
            routed_x,
            routed_s,
            routed_w.view(torch.uint8).reshape(-1, 4),
            valid.view(torch.uint8),
            route_map,
            P,
            A,
            E,
            int(topk),
            D,
            S,
            triton.next_power_of_2(D),
            triton.next_power_of_2(S),
            triton.next_power_of_2(E),
            4,
            num_warps=4,
        )
    return routed_x, routed_s, routed_w, valid
