"""CPU UT for ``combine_topk_swa_indices_cp_b1`` (Phase F2).

Mirrors the CP B=1 vectorized replacement of the Triton kernel
``_swa_ops_triton.combine_topk_swa_indices`` used by the workspace path
(``_attn_via_workspace`` for CSA / HCA layers under CP). Verifies that
per-rank Q rows produce the right per-token combined attention index
list when the kernel's ``pos = start_pos + token_idx_in_query`` formula
(which assumes contiguous Q) breaks under zigzag CP.

Standalone-loadable: stubs the ``rtp_llm`` package chain so the test
runs without ``libth_transformer_config.so`` being built (matches
``test_cp_context_build`` / ``test_cp_via_concat_meta`` /
``test_cp_indexer_seq_total`` / ``test_cp_topk_idxs_align``).
"""

import importlib.util
import os
import sys
import types

import torch


def _load_cp_module():
    if "rtp_llm.models_py.modules.dsv4.cp" in sys.modules:
        return sys.modules["rtp_llm.models_py.modules.dsv4.cp"]

    class _Group:
        TP = "TP"

    class _CT:
        Group = _Group

        @staticmethod
        def all_gather(*a, **k):
            return None

    for name in [
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
        "rtp_llm.models_py.modules",
        "rtp_llm.models_py.modules.dsv4",
    ]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["rtp_llm.models_py.distributed.collective_torch"] = _CT

    here = os.path.dirname(os.path.abspath(__file__))
    cp_path = os.path.normpath(os.path.join(here, os.pardir, "cp.py"))
    spec = importlib.util.spec_from_file_location(
        "rtp_llm.models_py.modules.dsv4.cp", cp_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules["rtp_llm.models_py.modules.dsv4.cp"] = mod
    return mod


_CP = _load_cp_module()
combine = _CP.combine_topk_swa_indices_cp_b1
combine_varlen = _CP.combine_topk_swa_indices_cp_varlen


def _kernel_reference(
    topk_indices: torch.Tensor,
    pos_per_token: torch.Tensor,
    seq_total: int,
    gather_len: int,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Pure-Python re-implementation of the Triton
    ``_combine_topk_swa_indices_kernel`` body for B==1 — used as the
    oracle. ``pos_per_token`` is the per-Q-row absolute global position
    (= ``cp_ctx.global_positions``); for non-CP / contiguous Q this is
    just ``arange(T) + start_pos``.
    """
    T = int(pos_per_token.shape[0])
    combined_topk = ((topk + window_size + 127) // 128) * 128
    out = torch.full((T, combined_topk), -1, dtype=torch.int32)
    lens = torch.zeros((T,), dtype=torch.int32)
    gather_start = seq_total - gather_len
    for t in range(T):
        pos = int(pos_per_token[t].item())
        topk_len = min((pos + 1) // compress_ratio, topk)
        swa_len = min(pos + 1, window_size)
        for off in range(topk_len):
            out[t, off] = topk_indices[t, off]
        for off in range(swa_len):
            out[t, topk_len + off] = N + off + pos - swa_len + 1 - gather_start
        lens[t] = topk_len + swa_len
    return out, lens


def _zigzag_global_positions(
    seq_len_full: int, cp_size: int, rank: int
) -> torch.Tensor:
    """Mirror the layout exercised in test_cp_context_build / cp.py:
    rank ``r`` owns first-half pair ``[r*pair, (r+1)*pair)`` then the
    mirrored second-half pair ``[full-(r+1)*pair, full-r*pair)``.
    """
    assert seq_len_full % (cp_size * 2) == 0
    pair = seq_len_full // (cp_size * 2)
    first = torch.arange(rank * pair, (rank + 1) * pair, dtype=torch.int64)
    second = torch.arange(
        seq_len_full - (rank + 1) * pair,
        seq_len_full - rank * pair,
        dtype=torch.int64,
    )
    return torch.cat([first, second])


def test_cp1_b1_fresh_matches_kernel_reference() -> None:
    """CP=1 fresh prefill (sp=0) — global_positions == arange(T).
    Helper output must equal the kernel-reference layout exactly.
    """
    seq = 64
    win = 16
    ratio = 4
    topk = 8
    sp = 0
    seq_total = sp + seq
    P = min(sp, win - 1)
    gather_len = seq + P
    N = seq_total // ratio
    M = N + gather_len

    topk_indices = torch.randint(0, max(1, N), (seq, topk), dtype=torch.int32)
    gp = torch.arange(seq, dtype=torch.int64)

    actual_idx, actual_lens = combine(
        topk_indices=topk_indices,
        global_positions=gp,
        sp_int=sp,
        window_size=win,
        compress_ratio=ratio,
        topk=topk,
        M=M,
        N=N,
    )
    expect_idx, expect_lens = _kernel_reference(
        topk_indices=topk_indices,
        pos_per_token=gp,
        seq_total=seq_total,
        gather_len=gather_len,
        window_size=win,
        compress_ratio=ratio,
        topk=topk,
        M=M,
        N=N,
    )
    assert torch.equal(actual_lens, expect_lens)
    assert torch.equal(actual_idx, expect_idx)


def test_cp2_b1_zigzag_each_rank_matches_kernel_at_global_pos() -> None:
    """CP=2 fresh prefill: each rank's helper output equals the
    kernel-reference computed at the rank's global positions. This is
    exactly what enables byte-equal CP=2 vs CP=1 attention output.
    """
    seq_full = 32
    cp_size = 2
    win = 16
    ratio = 4
    topk = 8
    sp = 0
    seq_total = sp + seq_full
    P = min(sp, win - 1)
    gather_len = seq_full + P
    N = seq_total // ratio
    M = N + gather_len

    for r in range(cp_size):
        gp = _zigzag_global_positions(seq_full, cp_size, r)
        topk_indices = torch.randint(
            0, max(1, N), (gp.shape[0], topk), dtype=torch.int32
        )

        actual_idx, actual_lens = combine(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=sp,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        expect_idx, expect_lens = _kernel_reference(
            topk_indices=topk_indices,
            pos_per_token=gp,
            seq_total=seq_total,
            gather_len=gather_len,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        assert torch.equal(actual_lens, expect_lens), f"rank {r}"
        assert torch.equal(actual_idx, expect_idx), f"rank {r}"


def test_cp2_b1_continuation_prefix_offset_threads_through() -> None:
    """Continuation prefill (sp > 0): SWA workspace slots are offset by
    ``gather_start = sp - P``; helper must consume sp_int and produce
    matching slots. Verified by oracle.
    """
    seq_full = 16
    cp_size = 2
    win = 8
    ratio = 4
    topk = 4
    sp = 24
    seq_total = sp + seq_full
    P = min(sp, win - 1)
    gather_len = seq_full + P
    N = seq_total // ratio
    M = N + gather_len

    gp_offset = sp
    for r in range(cp_size):
        rel = _zigzag_global_positions(seq_full, cp_size, r)
        gp = rel + gp_offset
        topk_indices = torch.randint(
            0, max(1, N), (gp.shape[0], topk), dtype=torch.int32
        )
        actual_idx, actual_lens = combine(
            topk_indices=topk_indices,
            global_positions=gp,
            sp_int=sp,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        expect_idx, expect_lens = _kernel_reference(
            topk_indices=topk_indices,
            pos_per_token=gp,
            seq_total=seq_total,
            gather_len=gather_len,
            window_size=win,
            compress_ratio=ratio,
            topk=topk,
            M=M,
            N=N,
        )
        assert torch.equal(actual_lens, expect_lens), f"rank {r}"
        assert torch.equal(actual_idx, expect_idx), f"rank {r}"


def test_cp_swa_only_topk0() -> None:
    """SWA-only layer (topk=0, ratio=1): no cmp slots, combined_lens
    collapses to swa_len. Mirrors ``_attn_fp8_swa_via_concat`` non-CP
    contract for the SWA-only via_concat path (Phase E shape).
    """
    seq = 16
    win = 8
    sp = 0
    P = min(sp, win - 1)
    gather_len = seq + P
    N = 0
    M = N + gather_len
    gp = torch.arange(seq, dtype=torch.int64)
    topk_indices = torch.zeros((seq, 0), dtype=torch.int32)
    actual_idx, actual_lens = combine(
        topk_indices=topk_indices,
        global_positions=gp,
        sp_int=sp,
        window_size=win,
        compress_ratio=1,
        topk=0,
        M=M,
        N=N,
    )
    # combined_lens = swa_len = min(gp+1, win)
    expected_lens = torch.minimum(gp + 1, gp.new_full((), win)).to(torch.int32)
    assert torch.equal(actual_lens, expected_lens)


def test_cp_varlen_b2_adds_request_workspace_offsets() -> None:
    win = 8
    ratio = 4
    topk = 4
    N = 6
    M = 32
    prefix = torch.tensor([0, 10], dtype=torch.int64)
    req = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    gp = torch.tensor([0, 7, 10, 13], dtype=torch.int64)
    topk_indices = torch.tensor(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ],
        dtype=torch.int32,
    )

    actual_idx, actual_lens = combine_varlen(
        topk_indices=topk_indices,
        global_positions=gp,
        sp_int=0,
        window_size=win,
        compress_ratio=ratio,
        topk=topk,
        M=M,
        N=N,
        req_id_per_token=req,
        prefix_lengths=prefix,
    )

    # Token 2 belongs to request 1, so both compressed and SWA entries are
    # shifted by one workspace stride.
    row = actual_idx[2]
    assert int(row[0].item()) == M + 0
    assert int(row[1].item()) == M + 1
    # gp=10, sp=10, P=7, gather_start=3, swa_len=8:
    # first SWA slot = M + N + (10 - 3) - 8 + 1 = M + N
    assert int(row[int(actual_lens[2].item()) - 8].item()) == M + N


def test_cp_varlen_b1_matches_b1_helper() -> None:
    seq = 16
    win = 8
    ratio = 4
    topk = 4
    sp = 0
    N = seq // ratio
    M = N + seq
    gp = torch.arange(seq, dtype=torch.int64)
    topk_indices = torch.randint(0, max(1, N), (seq, topk), dtype=torch.int32)
    idx_b1, lens_b1 = combine(
        topk_indices=topk_indices,
        global_positions=gp,
        sp_int=sp,
        window_size=win,
        compress_ratio=ratio,
        topk=topk,
        M=M,
        N=N,
    )
    idx_var, lens_var = combine_varlen(
        topk_indices=topk_indices,
        global_positions=gp,
        sp_int=sp,
        window_size=win,
        compress_ratio=ratio,
        topk=topk,
        M=M,
        N=N,
        req_id_per_token=torch.zeros(seq, dtype=torch.int32),
        prefix_lengths=torch.zeros(1, dtype=torch.int64),
    )
    assert torch.equal(lens_var, lens_b1)
    assert torch.equal(idx_var, idx_b1)


if __name__ == "__main__":
    test_cp1_b1_fresh_matches_kernel_reference()
    test_cp2_b1_zigzag_each_rank_matches_kernel_at_global_pos()
    test_cp2_b1_continuation_prefix_offset_threads_through()
    test_cp_swa_only_topk0()
    test_cp_varlen_b2_adds_request_workspace_offsets()
    test_cp_varlen_b1_matches_b1_helper()
    print("OK")
