"""Pure host-row reuse without caching CP geometry or validation results."""

from contextlib import ExitStack, contextmanager
import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_qkv as qkv
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import _plan
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_metadata_reuse import ENV, _fixture


@contextmanager
def _environment(values):
    # Never clear the process environment while native getenv threads run.
    absent = object()
    previous = {name: os.environ.get(name, absent) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, value in previous.items():
            if value is absent:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _reference_selected(lengths):
    result, offset = [], 0
    for length in lengths:
        result.extend(offset + row for row in range(max(0, length - 140), length))
        offset += length
    return tuple(result)


class CP8QueryProofReuseContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(_environment(ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch.object(torch.cuda, "is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch.object(torch.cuda, "current_stream",
                                             side_effect=AssertionError("CPU host proof must not initialize CUDA")))

    def test_geometry_reuses_only_the_same_lengths_selected_tuple(self):
        state, _ = _fixture()
        plan = _plan(state)
        before = qkv._cp8_selected_rows.cache_info()
        first = qkv._cp8_plan_geometry(plan)
        second = qkv._cp8_plan_geometry(plan)
        after = qkv._cp8_selected_rows.cache_info()
        self.assertEqual(after.hits - before.hits, 2)
        self.assertEqual(after.misses, before.misses)
        self.assertIsNot(first, second, "the complete geometry is rebuilt, never cached")
        self.assertEqual(first, second)
        self.assertIs(first[4], second[4])
        self.assertEqual(first[4], _reference_selected(state.lengths))
        self.assertEqual(len(first[4]), 1120)
        self.assertTrue(plan._ppu_cp8_query_birth.current())

    def test_selected_derivation_uses_lengths_values_and_has_a_bounded_cache(self):
        for lengths in ((4096,) * 8, (17, 140, 141, 256), (256, 272), (1, 2, 3)):
            with self.subTest(lengths=lengths):
                first = qkv._cp8_selected_rows(lengths)
                second = qkv._cp8_selected_rows(tuple(list(lengths)))
                self.assertIs(first, second)
                self.assertEqual(first, _reference_selected(lengths))
                self.assertIs(type(first), tuple)
        original = (4096,) * 8
        changed = (4112, *original[1:])
        first = qkv._cp8_selected_rows(original)
        next_rows = qkv._cp8_selected_rows(changed)
        self.assertIsNot(first, next_rows)
        self.assertEqual(next_rows, _reference_selected(changed))
        self.assertNotEqual(first, next_rows)
        for length in range(20001, 20066):
            self.assertEqual(qkv._cp8_selected_rows((length,)), _reference_selected((length,)))
        info = qkv._cp8_selected_rows.cache_info()
        self.assertEqual(info.maxsize, 64)
        self.assertLessEqual(info.currsize, 64)

    def test_new_valid_lengths_get_a_new_tuple_and_old_geometry_mutations_still_fail(self):
        state, _ = _fixture()
        plan = _plan(state)
        old = qkv._cp8_plan_geometry(plan)
        changed, _ = _fixture(lengths=(4112, *(4096,) * 7))
        other = _plan(changed)
        new = qkv._cp8_plan_geometry(other)
        self.assertIsNot(old[4], new[4])
        self.assertEqual(new[4], _reference_selected(changed.lengths))
        self.assertTrue(other._ppu_cp8_query_birth.current())
        for change in ("lengths", "prefixes", "rank", "chunks", "send_sizes", "replay"):
            with self.subTest(change=change):
                state, _ = _fixture()
                plan = _plan(state)
                birth = plan._ppu_cp8_query_birth
                geometry = qkv._cp8_plan_geometry(plan)
                self.assertTrue(birth.current())
                if change == "lengths":
                    plan.original_context.input_lengths_global_host = (4112, *(4096,) * 7)
                elif change == "prefixes":
                    plan.original_context.prefix_lengths_host = (126977, *(126976,) * 7)
                elif change == "rank":
                    plan.context.cp_rank = (plan.context.cp_rank + 1) % 8
                elif change == "chunks":
                    plan.context.chunk_lengths_per_req = (19, *(18,) * 7)
                elif change == "send_sizes":
                    plan.exchange.send_sizes = (plan.exchange.send_sizes[0] + 1, *plan.exchange.send_sizes[1:])
                else:
                    plan.context.swa_replay_starts_host = (0,) * 8
                self.assertFalse(birth.current())
                self.assertNotEqual(qkv._cp8_plan_geometry(plan), geometry)

    def test_transport_versions_clones_inference_replacements_and_borrowed_owners_remain_checked(self):
        for change in ("position_write", "mask_write", "receive_write", "receive_clone",
                       "cu_write", "inference_position", "borrowed_prefix"):
            with self.subTest(change=change):
                state, _ = _fixture()
                plan = _plan(state)
                birth = plan._ppu_cp8_query_birth
                geometry = qkv._cp8_plan_geometry(plan)
                self.assertTrue(birth.current())
                if change == "position_write":
                    plan.context.global_positions.add_(0)
                elif change == "mask_write":
                    plan.context.local_is_real.logical_not_().logical_not_()
                elif change == "receive_write":
                    plan.exchange.receive_positions.add_(0)
                elif change == "receive_clone":
                    plan.exchange.receive_positions = plan.exchange.receive_positions.clone()
                elif change == "cu_write":
                    plan.cu_seqlens.add_(0)
                elif change == "inference_position":
                    with torch.inference_mode():
                        plan.context.global_positions = plan.context.global_positions.clone()
                else:
                    plan.context.prefix_lengths = plan.context.prefix_lengths.clone()
                self.assertIs(qkv._cp8_plan_geometry(plan)[4], geometry[4])
                self.assertFalse(birth.current(), "a pure selected tuple hit must not certify mutated transport")


if __name__ == "__main__":
    unittest.main()
