"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import _write_pair_verdict


@triton.jit(do_not_specialize=["B"])
def _resolve_accepted_pair(
    ACCEPT,
    PREFIX,
    STATE_TABLE,
    CTX_TABLE,
    OUT,
    B,
    Q: tl.constexpr,
    PAGE: tl.constexpr,
    SC: tl.constexpr,
    SR: tl.constexpr,
    SS: tl.constexpr,
    CC: tl.constexpr,
    CR: tl.constexpr,
    CS: tl.constexpr,
    SP: tl.constexpr,
    CP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    live = row < B
    accepted = tl.load(ACCEPT + row, live, other=1).to(tl.int64)
    invalid = tl.sum((live & ((accepted < 1) | (accepted > Q))).to(tl.int32), 0) > 0
    # Clamped storage keeps speculative gathers in bounds; every clamped value
    # trips a verdict flag and aborts the commit before any pool write.
    safe_accepted = tl.minimum(tl.maximum(accepted, 1), Q)
    tl.store(OUT + 10 + 2 * B + row, safe_accepted, live)
    position = tl.load(PREFIX + row, live, other=0).to(tl.int64) + accepted - 1
    page = tl.maximum(position, 0) // PAGE
    state_block, state_invalid = _write_pair_verdict(
        STATE_TABLE, OUT, row, live, position, page, SC, SR, SS, SP, 2, 10
    )
    ctx_block, ctx_invalid = _write_pair_verdict(
        CTX_TABLE, OUT, row, live, position, page, CC, CR, CS, CP, 6, 10 + B
    )
    # Duplicate destinations inside one pool would double-write a physical
    # row. The two pools are distinct tensors, so cross-pool rows never
    # collide; compare within each list only.
    pair = (row[:, None] < row[None, :]) & live[:, None] & live[None, :]
    state_dup_mask = pair & (state_block[:, None] == state_block[None, :])
    state_dup = tl.sum(tl.sum(state_dup_mask.to(tl.int32), 1), 0) > 0
    ctx_dup_mask = pair & (ctx_block[:, None] == ctx_block[None, :])
    ctx_dup = tl.sum(tl.sum(ctx_dup_mask.to(tl.int32), 1), 0) > 0
    tl.store(OUT, invalid | state_invalid | ctx_invalid | state_dup | ctx_dup)
    tl.store(OUT + 1, invalid)
