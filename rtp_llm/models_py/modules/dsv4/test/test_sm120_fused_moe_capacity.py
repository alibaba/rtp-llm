"""CPU regression for the SM120 fused-MoE capture capacity contract.

The PD decode leg aborted with a raw native SIGABRT inside flashinfer
``cutlass_fused_moe`` on the FIRST ``fixed_ep`` CUDA-graph capture at batch>=2.
Root cause: the workspace AND ``tune_max_num_tokens`` were sized from the
PRE-gather per-rank budget ``cfg.max_tokens_per_rank``
(= ``max_generate_batch_size * (gen_num_per_cycle + 1)``), but on the fixed-EP
path the kernel actually processes the POST-gather tile of ``world * n_pad``
rows. For a 4-rank DP+EP MTP-3 config that tile is ``16 * capture_batch`` rows while the budget
is only ``4 * cap`` -- undersized whenever ``world * n_pad > budget``, which
flashinfer rejects by aborting natively (no Python exception => silent SIGABRT).

This test is pure-Python (no CUDA, no flashinfer, no dist). It pins the capacity
helper ``_sm120_fused_moe_capacity`` against the real decode-role token budget
``resolve_moe_max_tokens_per_rank`` for every recorded PD case, the single-rank
control, and the 512-row tiling boundary, and asserts the workspace cache key is
stable per capacity.
"""

from __future__ import annotations

import unittest

from rtp_llm.models_py.modules.dsv4.moe.moe_layer import (
    resolve_moe_max_tokens_per_rank,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4 import (
    _sm120_fused_moe_capacity,
)

# Fixed-EP tiling bound (deepep.py _forward_sm120_fixed_ep MAX_TILES) and the
# capture pad floor passed by the decode capture caller (pad_floor=4).
MAX_TILES = 512
PAD_FLOOR = 4
GEN_NUM_PER_CYCLE = 3  # DSpark MTP-3 => tokens_per_batch = gen_num_per_cycle + 1
TOKENS_PER_BATCH = GEN_NUM_PER_CYCLE + 1  # 4


def _decode_budget(cap: int) -> int:
    """The real decode-role per-rank token budget for a given admission cap."""
    return resolve_moe_max_tokens_per_rank(
        max_seq_len=32768,
        current_max_tokens_per_rank=8192,
        cp_size=1,
        max_generate_batch_size=cap,
        is_decode_role=True,
        is_speculative=True,
        gen_num_per_cycle=GEN_NUM_PER_CYCLE,
    )


def _post_gather_rows(world: int, capture_batch: int) -> int:
    """Rows the fused kernel sees: world * n_pad, n_pad = max(floor, bs*tokens)."""
    n = capture_batch * TOKENS_PER_BATCH
    n_pad = max(PAD_FLOOR, n)
    return world * n_pad


def _tile_cap(world: int, capture_batch: int) -> int:
    """The caller-declared max post-gather tile bound (min(MAX_TILES, total))."""
    return min(MAX_TILES, _post_gather_rows(world, capture_batch))


def _old_buggy_capacity(budget: int) -> int:
    """The pre-fix formula: sized from the pre-gather budget only."""
    return min(max(int(budget), 1), 512)


class DecodeBudgetTest(unittest.TestCase):
    def test_decode_budget_is_cap_times_tokens_per_batch(self):
        # MTP-3 decode role => budget = cap * 4 (the table's bound column).
        for cap, expected in ((1, 4), (2, 8), (4, 16), (16, 64)):
            self.assertEqual(_decode_budget(cap), expected, f"cap={cap}")


class CapacityCoversPostGatherTileTest(unittest.TestCase):
    """The four recorded PD cases (world=4, MTP-3) + the pass case."""

    # (cap, capture_batch, recorded_result)
    CASES = [
        (4, 1, "pass"),    # budget 16 == tile 16 -> the only case that booted
        (1, 1, "abort"),   # budget 4  <  tile 16
        (2, 2, "abort"),   # budget 8  <  tile 32
        (4, 4, "abort"),   # budget 16 <  tile 64
    ]

    def test_new_capacity_covers_every_recorded_case(self):
        for cap, bs, _ in self.CASES:
            budget = _decode_budget(cap)
            rows = _post_gather_rows(4, bs)
            tcap = _tile_cap(4, bs)
            cap_new = _sm120_fused_moe_capacity(budget, rows, tcap)
            self.assertGreaterEqual(
                cap_new, rows,
                f"cap={cap} bs={bs}: capacity {cap_new} must cover tile {rows}")

    def test_old_capacity_explains_the_aborts(self):
        # The pre-fix formula undersized exactly the aborting cases and matched
        # the one passing case -- this is the regression being fixed.
        for cap, bs, result in self.CASES:
            budget = _decode_budget(cap)
            rows = _post_gather_rows(4, bs)
            old = _old_buggy_capacity(budget)
            if result == "abort":
                self.assertLess(
                    old, rows,
                    f"cap={cap} bs={bs}: old capacity {old} should be < tile {rows}")
            else:
                self.assertGreaterEqual(
                    old, rows,
                    f"cap={cap} bs={bs}: old capacity {old} should cover tile {rows}")

    def test_capacity_equals_tile_for_decode_captures(self):
        # For decode capture the tile is < 512, so capacity == the tile exactly
        # (no over-allocation), and equals what the caller declared.
        for cap, bs, _ in self.CASES:
            rows = _post_gather_rows(4, bs)
            tcap = _tile_cap(4, bs)
            cap_new = _sm120_fused_moe_capacity(_decode_budget(cap), rows, tcap)
            self.assertEqual(cap_new, rows)
            self.assertEqual(tcap, rows)


class SingleRankControlTest(unittest.TestCase):
    def test_world1_never_undersized(self):
        # world=1 (LocalLoop / single card): tile == n == budget when cap==bs,
        # so the path was never exposed to the bug; the fix must not change it.
        for cap in (1, 2, 4):
            budget = _decode_budget(cap)
            rows = _post_gather_rows(1, cap)  # capture_batch == cap
            tcap = _tile_cap(1, cap)
            cap_new = _sm120_fused_moe_capacity(budget, rows, tcap)
            self.assertGreaterEqual(cap_new, rows)
            self.assertEqual(cap_new, max(rows, budget))


class TilingBoundaryTest(unittest.TestCase):
    def test_prefill_scale_tiles_share_one_512_capacity(self):
        # A prefill-fallback-sized gather (total_rows > 512) tiles at 512; every
        # tile (including the short last one) must be covered, and all tiles map
        # to the SAME capacity => one stable workspace buffer / cache entry.
        world, n_pad = 4, 250
        total_rows = world * n_pad  # 1000
        self.assertGreater(total_rows, MAX_TILES)
        tcap = min(MAX_TILES, total_rows)
        self.assertEqual(tcap, MAX_TILES)
        budget = 4096  # a prefill-scale per-rank budget
        caps = set()
        for offset in range(0, total_rows, MAX_TILES):
            end = min(offset + MAX_TILES, total_rows)
            rows = end - offset
            cap_new = _sm120_fused_moe_capacity(budget, rows, tcap)
            self.assertGreaterEqual(cap_new, rows, f"tile rows={rows}")
            caps.add(cap_new)
        # Full tiles clamp to the 512 budget-derived ceiling; capacity is stable.
        self.assertEqual(caps, {MAX_TILES})

    def test_capacity_never_below_budget_floor(self):
        # A tiny tile still gets at least the budget-derived floor (preserves the
        # autotuner bucket range for the non-fixed-EP / warmup callers).
        self.assertEqual(_sm120_fused_moe_capacity(64, 4, 4), 64)
        self.assertEqual(_sm120_fused_moe_capacity(4096, 4, 4), 512)

    def test_zero_and_symbolic_rows_fall_back_to_budget(self):
        # int(x.shape[0]) is guarded; a symbolic/unknown row count (0) must not
        # shrink capacity below the caller bound or the budget.
        self.assertEqual(_sm120_fused_moe_capacity(16, 0, 32), 32)
        self.assertEqual(_sm120_fused_moe_capacity(16, 0, 0), 16)


class CacheKeyStabilityTest(unittest.TestCase):
    def test_distinct_capture_batches_yield_distinct_capacities(self):
        # The workspace cache key carries max_tokens; distinct capture batches
        # (16/32/64) must produce distinct capacities => distinct stable buffers.
        caps = [
            _sm120_fused_moe_capacity(_decode_budget(4), _post_gather_rows(4, bs),
                                      _tile_cap(4, bs))
            for bs in (1, 2, 4)
        ]
        self.assertEqual(caps, [16, 32, 64])
        self.assertEqual(len(set(caps)), 3)

    def test_same_capacity_is_deterministic(self):
        a = _sm120_fused_moe_capacity(16, 64, 64)
        b = _sm120_fused_moe_capacity(16, 64, 64)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
