"""Pure-CPU tests for decode backend selection policies."""

import unittest

from rtp_llm.models_py.modules.factory.attention.dispatch.selector import (
    kv_grid,
    select_stable,
)


# kv grid
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


# stable production selection
def test_select_stable_empty_matrix_returns_none():
    assert select_stable({}, ["A", "B"], "A") is None


def test_select_stable_keeps_default_below_threshold():
    matrix = {"A": [100.0], "B": [96.0]}
    assert select_stable(matrix, ["A", "B"], "A", threshold=0.05) == "A"


def test_select_stable_switches_for_significant_improvement():
    matrix = {"A": [100.0], "B": [90.0]}
    assert select_stable(matrix, ["A", "B"], "A", threshold=0.05) == "B"


def test_select_stable_uses_registry_order_within_cluster():
    matrix = {"A": [100.0], "B": [102.0]}
    assert select_stable(matrix, ["B", "A"], cluster_margin=0.05) == "B"


def test_select_stable_clearly_fastest_wins_over_registry_priority():
    matrix = {"A": [100.0], "B": [120.0]}
    assert select_stable(matrix, ["B", "A"], cluster_margin=0.05) == "A"


def test_select_stable_works_when_default_is_not_measured():
    matrix = {"A": [100.0], "B": [90.0]}
    assert select_stable(matrix, ["A", "B"], "default") == "B"


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
