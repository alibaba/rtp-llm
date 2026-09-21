import importlib.util
from pathlib import Path
import random
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "k3_chunk_plan",
    Path(__file__).resolve().parents[2]
    / "rtp_llm/models_py/modules/kimi_k3/chunk_plan.py",
)
planner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = planner
spec.loader.exec_module(planner)


def verify_coverage(lengths, prefixes, budget, block):
    rounds = planner.plan_kimi_k3_chunk_rounds(
        lengths, prefixes, chunk_budget=budget, alignment_tokens=block
    )
    consumed = [0] * len(lengths)
    terminal_count = [0] * len(lengths)
    offsets = [sum(lengths[:i]) for i in range(len(lengths))]
    for chunk in rounds:
        assert 0 < chunk.token_count <= budget
        assert len({s.original_batch_idx for s in chunk.slices}) == len(chunk.slices)
        for s in chunk.slices:
            i = s.original_batch_idx
            assert s.processed_length == consumed[i]
            assert s.source_start == offsets[i] + consumed[i]
            assert s.source_end - s.source_start == s.new_length > 0
            assert s.absolute_start == prefixes[i] + consumed[i]
            consumed[i] += s.new_length
            assert s.absolute_end == prefixes[i] + consumed[i]
            assert s.terminal == (consumed[i] == lengths[i])
            terminal_count[i] += int(s.terminal)
            if not s.terminal:
                assert s.absolute_end % block == 0
    assert consumed == lengths
    assert terminal_count == [1] * len(lengths)
    return rounds


@pytest.mark.parametrize("extra", [1, 7])
@pytest.mark.parametrize("prefix", [0, 4096, 65536])
def test_exact_budget_tail_with_cold_and_reused_prefix(extra, prefix):
    rounds = verify_coverage([65536 + extra], [prefix], 65536, 4096)
    assert [r.token_count for r in rounds] == [65536, extra]
    assert rounds[1].slices[0].absolute_start == prefix + 65536


def test_alignment_is_ordinary_block_not_tp_times_block():
    rounds = verify_coverage([4097, 8193], [4096, 8192], 8192, 4096)
    assert rounds[0].slices[0].terminal
    assert rounds[1].slices[0].absolute_end == 16384


def test_random_mixed_batch_rounds_have_no_dropped_or_duplicated_tokens():
    rng = random.Random(327)
    for batch in [1, 2, 3, 7, 8, 9]:
        for _ in range(30):
            lengths = [rng.randrange(1, 200000) for _ in range(batch)]
            prefixes = [rng.randrange(0, 30) * 4096 for _ in range(batch)]
            verify_coverage(lengths, prefixes, 65536, 4096)


@pytest.mark.parametrize(
    "lengths,prefixes,budget,block",
    [
        ([], [], 65536, 4096),
        ([1], [], 65536, 4096),
        ([0], [0], 65536, 4096),
        ([10], [-4096], 65536, 4096),
        ([10], [1], 65536, 4096),
        ([10], [0], 0, 4096),
        ([10], [0], 65536, 0),
        ([8192], [0], 1024, 4096),
    ],
)
def test_invalid_plan_fails_before_any_execution(lengths, prefixes, budget, block):
    with pytest.raises(ValueError):
        planner.plan_kimi_k3_chunk_rounds(
            lengths, prefixes, chunk_budget=budget, alignment_tokens=block
        )
