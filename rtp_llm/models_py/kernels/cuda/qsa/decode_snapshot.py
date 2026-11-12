"""Device kernels; model state and execution policy belong to the caller."""

import triton
import triton.language as tl


@triton.jit
def _snapshot_decode(
    POS,
    KV,
    KT,
    STATE,
    ST,
    SS,
    OS,
    KS,
    OK,
    KTW: tl.constexpr,
    STW: tl.constexpr,
    KB: tl.constexpr,
    SB: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, 128)
    # Match the reference's int64 conversion before arithmetic, even for
    # malformed int32 positions inspected by standalone validation.
    pos = tl.load(POS + row).to(tl.int64)
    sc = tl.minimum(tl.maximum(pos >> 7, 0), STW - 1)
    kc = tl.minimum(tl.maximum(pos >> 7, 0), KTW - 1)
    sid = tl.load(ST + row * STW + sc).to(tl.int64)
    kid = tl.load(KT + row * KTW + kc).to(tl.int64)
    ss = tl.minimum(tl.maximum(sid * 8 + (pos & 7), 0), SB * 8 - 1)
    ks = tl.where(((pos + 1) & 3) == 0, kid * 32 + ((pos >> 2) & 31), 0)
    ks = tl.minimum(tl.maximum(ks, 0), KB * 32 - 1)
    state_live = (pos >= 0) & (pos >> 7 < STW) & (sid > 0) & (sid < SB)
    kv_live = (
        (pos >= 0) & (pos >> 7 < KTW) & (kid > 0) & (kid < KB) & (((pos + 1) & 3) == 0)
    )
    tl.store(SS + row, tl.where(state_live, ss, -1))
    tl.store(KS + row, tl.where(kv_live, ks, -1))
    old_state = tl.load(STATE + ss * 128 + d)
    old_kv = tl.load(KV + ks * 128 + d)
    tl.store(OS + row * 128 + d, old_state)
    tl.store(OK + row * 128 + d, old_kv)
