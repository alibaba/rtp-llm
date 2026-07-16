"""Unit test for CP local KV length helpers (CP-sharded reader fix).

Regression for the rank-1 NaN bug discovered in
``v4_flash_pd_cp2ep2_dp2ep2_kv_cache_sharded_sm100``: the sharded-pool
reader was passing the PADDED per-rank length to ``dequantize_and_gather_k_cache``,
which over-read past the rank's actually-written entries and dequanted
uninitialized FP8 bytes (NaN). The fix is to pass the per-rank ACTUAL
owned KV count instead. This test pins the formula.
"""

from __future__ import annotations

import sys

import torch

from rtp_llm.models_py.modules.dsv4.cp import (
    cp_actual_owned_kv_lens as _compute_local_owned_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.cp import (
    cp_padded_local_kv_lens as _compute_local_seq_lens,
)


def _ref_owned_count(T: int, cp_size: int, block_size: int, cp_rank: int) -> int:
    """Reference impl: enumerate logical blocks owned by cp_rank, sum tokens."""
    total_blocks = (T + block_size - 1) // block_size
    n = 0
    for g_blk in range(total_blocks):
        if g_blk % cp_size != cp_rank:
            continue
        start = g_blk * block_size
        end = min(start + block_size, T)
        n += end - start
    return n


def test_local_owned_matches_reference():
    for cp_size in (1, 2, 4):
        for T in (
            0,
            1,
            63,
            64,
            65,
            127,
            128,
            191,
            256,
            320,
            384,
            511,
            512,
            691,
            1024,
            4096,
        ):
            per_req = torch.tensor([T], dtype=torch.int64)
            for cp_rank in range(cp_size):
                got = int(
                    _compute_local_owned_kv_lens(per_req, cp_size, 64, cp_rank)[
                        0
                    ].item()
                )
                ref = _ref_owned_count(T, cp_size, 64, cp_rank)
                assert got == ref, (
                    f"T={T} cp_size={cp_size} cp_rank={cp_rank}: "
                    f"got={got} ref={ref}"
                )


def test_owned_sum_equals_total():
    """Sum of owned counts across all ranks must equal T (no double-count, no leak)."""
    block_size = 64
    for cp_size in (1, 2, 4):
        for T in (0, 1, 64, 100, 691, 1024):
            per_req = torch.tensor([T], dtype=torch.int64)
            total = sum(
                int(
                    _compute_local_owned_kv_lens(per_req, cp_size, block_size, r)[
                        0
                    ].item()
                )
                for r in range(cp_size)
            )
            assert total == T, f"cp_size={cp_size} T={T}: sum_owned={total} != T={T}"


def test_owned_le_padded():
    """Owned count must never exceed the padded local length."""
    block_size = 64
    for cp_size in (1, 2, 4):
        for T in (0, 64, 65, 691, 1024):
            per_req = torch.tensor([T], dtype=torch.int64)
            padded = int(
                _compute_local_seq_lens(per_req, cp_size, block_size)[0].item()
            )
            for r in range(cp_size):
                owned = int(
                    _compute_local_owned_kv_lens(per_req, cp_size, block_size, r)[
                        0
                    ].item()
                )
                assert (
                    owned <= padded
                ), f"cp_size={cp_size} cp_rank={r} T={T}: owned={owned} > padded={padded}"


def test_regression_smoke_case():
    """The exact case from the failing smoke: T=691, cp_size=2, block_size=64.
    Rank 0 owns g_blk={0,2,4,6,8,10}: 5 full + partial(51) = 371 entries.
    Rank 1 owns g_blk={1,3,5,7,9}: 5 full = 320 entries.
    Padded local len = ceil(691 / 128) * 64 = 6 * 64 = 384 (uniform across ranks).
    """
    per_req = torch.tensor([691], dtype=torch.int64)
    assert int(_compute_local_owned_kv_lens(per_req, 2, 64, 0)[0].item()) == 371
    assert int(_compute_local_owned_kv_lens(per_req, 2, 64, 1)[0].item()) == 320
    assert int(_compute_local_seq_lens(per_req, 2, 64)[0].item()) == 384


def test_multi_request_batch():
    """Batched per-request lengths must each independently obey the formula."""
    per_req = torch.tensor([0, 64, 65, 691, 1024], dtype=torch.int64)
    cp_size, block_size = 2, 64
    for cp_rank in range(cp_size):
        got = _compute_local_owned_kv_lens(per_req, cp_size, block_size, cp_rank)
        for i, T in enumerate(per_req.tolist()):
            assert int(got[i].item()) == _ref_owned_count(
                T, cp_size, block_size, cp_rank
            )


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
