"""selector pure-CPU unit tests: capture buckets / kv grid / aggregation criteria / two-stage filtering + fallback orchestration.

GPU touchpoints (eligible/measure/fallback) all use injected lambdas, do not touch
torch/rtp_llm, and run on GPU-less machines / in CI. The GPU wiring layer
run_backend_selection is not tested here (it needs a real machine).
"""

import unittest
from contextlib import contextmanager

from rtp_llm.models_py.modules.factory.attention.dispatch.selector import (
    capture_buckets,
    kv_grid,
    select_min_mean,
    select_minimax_regret,
    select_plan,
)


@contextmanager
def _assert_raises(exc):
    """Minimal stand-in for pytest.raises (bazel py_test has no pytest)."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} to be raised")


# ─── capture bucket rules ────────────────────────────────
def test_capture_buckets_default_rule():
    assert capture_buckets(32) == [1, 2, 4, 8, 16, 24, 32]
    assert capture_buckets(64) == [1, 2, 4, 8, 16, 24, 32, 48, 64]
    assert capture_buckets(80) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 80]


def test_capture_buckets_covers_128():
    assert capture_buckets(128) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]


def test_capture_buckets_beyond_128_steps():
    assert capture_buckets(200) == [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        144,
        160,
        176,
        192,
        200,
    ]


def test_capture_buckets_appends_max_when_off_step():
    assert capture_buckets(70) == [1, 2, 4, 8, 16, 24, 32, 48, 64, 70]


def test_capture_buckets_small():
    assert capture_buckets(1) == [1]
    assert capture_buckets(10) == [1, 2, 4, 8, 10]


def test_capture_buckets_invalid():
    with _assert_raises(ValueError):
        capture_buckets(0)


# ─── kv grid ─────────────────────────────────────────────────────────────────
def test_kv_grid_clips_to_max_seq_len():
    assert kv_grid(8192) == [256, 512, 1024, 2048, 4096, 8192]


def test_kv_grid_at_64k():
    assert kv_grid(65536) == [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]


def test_kv_grid_full_at_256k():
    assert kv_grid(262144) == [
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
    ]


def test_kv_grid_appends_tail_boundary():
    g = kv_grid(40000)
    assert g[-1] == 40000
    assert g[:-1] == [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def test_kv_grid_degenerate_small():
    assert kv_grid(100) == [100]


# ─── aggregation criteria ─────────────────────────────────────────────────────────────────
def test_select_min_mean_picks_lowest_average():
    matrix = {"A": [10.0, 10.0, 100.0], "B": [30.0, 30.0, 30.0]}
    assert select_min_mean(matrix) == "B"


def test_select_min_mean_stable_tie():
    matrix = {"A": [10.0, 20.0], "B": [20.0, 10.0]}
    assert select_min_mean(matrix) == "A"


def test_select_min_mean_empty_none():
    assert select_min_mean({}) is None


def test_select_minimax_regret_prefers_robust():
    matrix = {"A": [1.0, 110.0], "B": [2.0, 100.0]}
    assert select_minimax_regret(matrix) == "A"
    assert select_min_mean(matrix) == "B"


def test_select_minimax_regret_matches_mean_when_dominant():
    matrix = {"A": [10.0, 10.0], "B": [20.0, 20.0]}
    assert select_minimax_regret(matrix) == "A"
    assert select_min_mean(matrix) == "A"


def test_select_minimax_regret_empty_none():
    assert select_minimax_regret({}) is None


# ─── select_plan orchestration ─────────────────────────────────────────────────────────
def _no_fallback(_bs):
    return None


def test_select_plan_picks_best_per_bucket():
    grid = [256, 4096]
    lat = {
        (1, "A"): [50.0, 50.0],
        (1, "B"): [20.0, 20.0],
        (8, "A"): [10.0, 10.0],
        (8, "B"): [40.0, 40.0],
    }
    plan = select_plan(
        [1, 8],
        eligible_fn=lambda bs: ["A", "B"],
        measure_fn=lambda impl, bs, kv: lat[(bs, impl)][grid.index(kv)],
        fallback_fn=_no_fallback,
        grid=grid,
    )
    assert plan.backend_for(1) == "B"
    assert plan.backend_for(8) == "A"


def test_select_plan_empty_eligible_falls_back():
    plan = select_plan(
        [16],
        eligible_fn=lambda bs: [],
        measure_fn=lambda i, b, k: 1.0,
        fallback_fn=lambda bs: "XQAImpl",
        grid=[256],
    )
    assert plan.backend_for(16) == "XQAImpl"


def test_select_plan_all_na_falls_back():
    plan = select_plan(
        [8],
        eligible_fn=lambda bs: ["A", "B"],
        measure_fn=lambda i, b, k: None,
        fallback_fn=lambda bs: "PyFlashinferDecodeImpl",
        grid=[256, 4096],
    )
    assert plan.backend_for(8) == "PyFlashinferDecodeImpl"


def test_select_plan_partial_na_drops_that_impl():
    def measure_fn(impl, bs, kv):
        if impl == "A":
            return 1.0 if kv == 256 else None
        return 10.0

    plan = select_plan(
        [8],
        eligible_fn=lambda bs: ["A", "B"],
        measure_fn=measure_fn,
        fallback_fn=_no_fallback,
        grid=[256, 4096],
    )
    assert plan.backend_for(8) == "B"


def test_select_plan_measure_exception_treated_as_na():
    def measure_fn(impl, bs, kv):
        raise RuntimeError("boom")

    plan = select_plan(
        [8],
        eligible_fn=lambda bs: ["A"],
        measure_fn=measure_fn,
        fallback_fn=lambda bs: "XQAImpl",
        grid=[256],
    )
    assert plan.backend_for(8) == "XQAImpl"


def test_select_plan_no_fallback_leaves_bucket_unassigned():
    plan = select_plan(
        [8],
        eligible_fn=lambda bs: [],
        measure_fn=lambda i, b, k: 1.0,
        fallback_fn=_no_fallback,
        grid=[256],
    )
    assert plan.backend_for(8) is None
    assert plan.buckets() == []


def test_select_plan_minimax_selector_injectable():
    grid = [256, 4096]
    lat = {"A": [1.0, 110.0], "B": [2.0, 100.0]}

    def _run(selector):
        return select_plan(
            [4],
            eligible_fn=lambda bs: ["A", "B"],
            measure_fn=lambda i, b, k: lat[i][grid.index(k)],
            fallback_fn=_no_fallback,
            grid=grid,
            selector=selector,
        ).backend_for(4)

    assert _run(select_minimax_regret) == "A"
    assert _run(select_min_mean) == "B"


# Bind the module-level test_* functions onto a TestCase so bazel's unittest
# runner (no pytest available) discovers and runs them.
class SelectorTest(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith("test_") and callable(_fn):
        setattr(SelectorTest, _name, staticmethod(_fn))
del _name, _fn  # don't leak a class/func ref that unittest would re-collect


if __name__ == "__main__":
    unittest.main()
