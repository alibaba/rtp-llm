"""Mainline result semantics for the five retained manual PPU scenarios."""

import os
from typing import Optional


MAINLINE_PPU_PERF_TESTS = frozenset(
    {
        "qwen35_moe_ppu_decode_dp8_ll_cg_full_grid",
        "qwen35_moe_ppu_decode_dp8_seq65k",
        "qwen35_moe_ppu_decode_dp8_seq32k",
        "qwen35_moe_ppu_decode_dp8_ll_cg_kspb128",
        "qwen35_moe_ppu_prefill_tp8_deepep_normal_deepgemm_contiguous_v3",
    }
)


def uses_mainline_perf_semantics(test_name: Optional[str] = None) -> bool:
    """Use exact shared Bazel/Native names; other benchmarks keep their policy."""
    if test_name is None:
        test_name = os.environ.get("PERF_TEST_NAME", "")
    return test_name in MAINLINE_PPU_PERF_TESTS
