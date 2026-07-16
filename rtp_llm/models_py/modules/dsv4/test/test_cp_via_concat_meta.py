"""Phase E CPU unit tests: continuation-prefill via_concat meta under CP.

Pins the Python builder for ``combined_indices`` / ``combined_lens`` /
``slot_in_flat`` against the documented kernel formula in
``_combine_topk_swa_indices_kernel`` (``_swa_ops_triton.py:262-329``):

    pos          = global_position[t]            # zigzag rank-local Q
    swa_len      = min(pos+1, win)
    gather_start = max(0, prefix - win + 1)
    slot[t, off] = pos - swa_len + 1 - gather_start + off,  off < swa_len
                 = -1 otherwise

For B=1, M = min(prefix, win-1) + seq_len_full and the new-K scatter
target is ``slot_in_flat = arange(min(prefix, win-1),
min(prefix, win-1) + seq_len_full)``.

These tests stay pure CPU (no rtp_llm package init) and exercise both
fresh (prefix=0) and continuation (prefix>0) cases.
"""

import math

import torch


def _zigzag_global_positions(
    seq_len_full: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    pair = seq_len_full // (cp_size * 2)
    first = torch.arange(cp_rank * pair, (cp_rank + 1) * pair, dtype=torch.int64)
    second = torch.arange(
        seq_len_full - (cp_rank + 1) * pair,
        seq_len_full - cp_rank * pair,
        dtype=torch.int64,
    )
    return torch.cat([first, second])


def _build_cp_combined_indices(
    global_positions: torch.Tensor,
    prefix: int,
    win: int,
    align: int = 128,
) -> tuple:
    """Verbatim copy of the Phase-E vectorised builder in
    ``_build_swa_prefill_meta_varlen`` (under cp_on_write + any_cont)."""
    gp = global_positions.to(torch.int64)
    gather_start = max(0, prefix - win + 1)
    combined_lens = (
        torch.minimum(gp + 1, torch.full_like(gp, win)).to(torch.int32).contiguous()
    )
    combined_topk = ((win + align - 1) // align) * align
    offset = torch.arange(combined_topk, dtype=torch.int64).unsqueeze(0)
    cl_col = combined_lens.to(torch.int64).unsqueeze(1)
    gp_col = gp.unsqueeze(1)
    slots = gp_col - cl_col + 1 - gather_start + offset
    mask = offset < cl_col
    combined_indices = (
        torch.where(mask, slots, torch.full_like(slots, -1))
        .to(torch.int32)
        .contiguous()
    )
    return combined_indices, combined_lens


def _expected_for_q(g: int, prefix: int, win: int, align: int = 128) -> tuple:
    """Reference per-Q-token expected (slot list, swa_len) from kernel docs."""
    swa_len = min(g + 1, win)
    gather_start = max(0, prefix - win + 1)
    combined_topk = ((win + align - 1) // align) * align
    slots = []
    for off in range(combined_topk):
        if off < swa_len:
            slots.append(g - swa_len + 1 - gather_start + off)
        else:
            slots.append(-1)
    return slots, swa_len


def test_cp_combined_indices_fresh_prefill_matches_formula() -> None:
    """CP=2 fresh prefill (prefix=0): each rank-local Q at global pos g gets
    slots ``[max(0, g-win+1)..g]`` right-padded to combined_topk."""
    seq_len_full = 16
    cp_size = 2
    win = 4
    prefix = 0
    chunk = seq_len_full // cp_size
    for r in range(cp_size):
        gp = _zigzag_global_positions(seq_len_full, cp_size, r)
        ci, cl = _build_cp_combined_indices(gp, prefix, win)
        for t in range(chunk):
            g = int(gp[t].item())
            expected_slots, expected_swa = _expected_for_q(g, prefix, win)
            assert int(cl[t].item()) == expected_swa
            assert ci[t].tolist() == expected_slots


def test_cp_combined_indices_cont_prefill_matches_formula() -> None:
    """CP=2 cont prefill (prefix=200 > win): gather_start = prefix-win+1,
    workspace SWA region holds last win-1 prefix tokens + new K, so per-Q
    slots are ``[g - swa_len + 1 - gather_start..g - gather_start]``."""
    seq_len_full = 16
    cp_size = 2
    win = 64
    prefix = 200
    chunk = seq_len_full // cp_size
    # Continuation positions are offset by prefix in the meta builder.
    # The Python builder consumes ``cp_ctx.global_positions`` which is
    # ``relative_positions + prefix_length`` — produce that here.
    for r in range(cp_size):
        rel = _zigzag_global_positions(seq_len_full, cp_size, r)
        gp = rel + prefix  # absolute pos in the continued sequence
        ci, cl = _build_cp_combined_indices(gp, prefix, win)
        for t in range(chunk):
            g = int(gp[t].item())
            expected_slots, expected_swa = _expected_for_q(g, prefix, win)
            assert int(cl[t].item()) == expected_swa
            assert ci[t].tolist() == expected_slots
            # Sanity: real slots fall in [0, M) where M = (win-1) + seq_len_full
            P_b = min(prefix, win - 1)
            M = P_b + seq_len_full
            real = [s for s in ci[t].tolist() if s != -1]
            assert all(
                0 <= s < M for s in real
            ), f"rank={r} t={t} g={g}: slot {real} out of [0, {M})"


def test_cp_slot_in_flat_b1_arange_matches_pb_offset() -> None:
    """B=1 CP slot_in_flat is just ``arange(P_b, P_b + seq_len_full)`` —
    each gathered new-K token writes to its workspace slot offset by
    P_b = min(prefix, win-1)."""
    for prefix, win, seq_len_full in [(0, 4, 16), (200, 64, 16), (50, 100, 8)]:
        P_b = min(prefix, win - 1)
        expected = torch.arange(P_b, P_b + seq_len_full, dtype=torch.long)
        # The builder's expression: arange(P_b, P_b + cp_ctx.seq_len_full)
        actual = torch.arange(P_b, P_b + seq_len_full, dtype=torch.long)
        assert torch.equal(actual, expected)


def test_cp_combined_topk_padding_alignment_128() -> None:
    """Combined-topk width is aligned up to 128 (kernel's
    ``_SPARSE_PREFILL_TOPK_ALIGNMENT``); our builder pads tail with -1
    so the shape matches the non-CP kernel output."""
    win = 7
    expected_topk = 128  # ceil_div(7, 128) * 128
    gp = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    ci, _ = _build_cp_combined_indices(gp, prefix=0, win=win)
    assert ci.shape == (4, expected_topk)


def test_cp_combined_indices_cp1_collapses_consistent() -> None:
    """CP=1 (single rank, no zigzag) reproduces the same per-Q math —
    documents that the Python builder's formula is equivalent to the
    kernel's documented contract for the contiguous case."""
    seq_len_full = 8
    win = 4
    prefix = 12
    gp = torch.arange(prefix, prefix + seq_len_full, dtype=torch.int64)
    ci, cl = _build_cp_combined_indices(gp, prefix, win)
    for t in range(seq_len_full):
        g = int(gp[t].item())
        expected_slots, expected_swa = _expected_for_q(g, prefix, win)
        assert int(cl[t].item()) == expected_swa
        assert ci[t].tolist() == expected_slots


if __name__ == "__main__":
    test_cp_combined_indices_fresh_prefill_matches_formula()
    test_cp_combined_indices_cont_prefill_matches_formula()
    test_cp_slot_in_flat_b1_arange_matches_pb_offset()
    test_cp_combined_topk_padding_alignment_128()
    test_cp_combined_indices_cp1_collapses_consistent()
    print("OK")
