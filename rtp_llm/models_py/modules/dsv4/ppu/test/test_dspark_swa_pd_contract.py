"""CPU contract for the DSpARK draft SWA ring slots and PD KV handoff.

The draft's non-causal query block writes ``gamma`` rows into a per-block
sliding-window ring and then reads ``window + gamma`` slots in one pass, while
the commit path writes the committed rows through
``compute_swa_slot_mapping_from_positions``.  If those two ever disagree, or if
the ring stops holding ``window + gamma`` distinct slots, the proposal reads
uninitialised pool entries and silently emits garbage.  Importing the model
pulls in the native runtime, so the two pure-python helpers are exec'd straight
from source here and exercised with real CPU torch.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple

import torch

_SRC = (
    Path(__file__).resolve().parents[6]
    / "rtp_llm/models_py/model_desc/deepseek_v4_dspark_model.py"
)
_WINDOW = 128
_GAMMA = 5
# window + gamma rounded up to an even count (C++ OpaqueKVCacheSpec::stateRingEntries).
_RING = 134
_TOKENS_PER_BLOCK = 256  # DSV4_TOKENS_PER_BLOCK
_TOP_INDEX_WIDTH = 256  # RICH_SWA_MAX_INDEX_WIDTH


def _load(cls_name, names):
    tree = ast.parse(_SRC.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    nodes = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []  # staticmethod -> plain function
            nodes.append(node)
    found = {n.name for n in nodes}
    assert found == set(names), f"helpers moved or renamed: {found} != {set(names)}"
    # The source relies on ``from __future__ import annotations``; the synthetic
    # module here does not, so provide the names its signatures reference.
    ns = {"torch": torch, "Tuple": Tuple, "Optional": Optional, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SRC), "exec"), ns)
    return [ns[name] for name in names]


_GLOBAL_POOL_SLOTS, _BUILD_NONCAUSAL = _load(
    "DeepSeekV4DSparkModel", ("_global_pool_slots", "_build_noncausal_indices")
)


def _load_function(name):
    tree = ast.parse(_SRC.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    ns = {"Tuple": Tuple, "Optional": Optional, "Any": Any}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(_SRC), "exec"), ns)
    return ns[name]


_REQUIRE_RING = _load_function("_require_swa_ring_capacity")


def _model(window=_WINDOW, gamma=_GAMMA):
    """Instance-shaped stub exposing only what the two helpers touch."""

    return SimpleNamespace(
        _gen_num_per_cycle=gamma,
        _v4_args=SimpleNamespace(window_size=window),
        _global_pool_slots=_GLOBAL_POOL_SLOTS,
    )


def _commit_slots(block_ids, positions, seq_len, ring=_RING, tpb=_TOKENS_PER_BLOCK):
    """Mirror ``_compute_swa_slot_mapping_from_positions_kernel`` for one request.

    Only the final ``ring`` positions before ``min(block_end, seq_len)`` are
    written; older generations of the same ring slot map to ``-1``.
    """

    slots = []
    for pos, block_id in zip(positions, block_ids):
        block_in_seq = pos // tpb
        effective_end = min((block_in_seq + 1) * tpb, seq_len)
        tail_write = (pos + ring) >= effective_end
        slots.append(block_id * ring + (pos % ring) if (block_id > 0 and tail_write) else -1)
    return torch.tensor(slots, dtype=torch.long)


class DsparkSwaRingGeometryTest(unittest.TestCase):
    def test_ring_smaller_than_window_plus_gamma_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SWA ring is too small"):
            _REQUIRE_RING(_WINDOW, _GAMMA, _WINDOW + _GAMMA - 1)

    def test_ring_at_threshold_is_accepted(self):
        _REQUIRE_RING(_WINDOW, _GAMMA, _WINDOW + _GAMMA)  # no raise
        slots, length = _BUILD_NONCAUSAL(
            _model(),
            torch.tensor([200]),
            torch.tensor([True]),
            torch.tensor([[3, 4]]),
            _WINDOW + _GAMMA,
            _TOKENS_PER_BLOCK,
        )
        self.assertEqual(slots.dtype, torch.int32)
        self.assertEqual(tuple(slots.shape), (_GAMMA, _TOP_INDEX_WIDTH))
        self.assertEqual(int(length[0]), _WINDOW + _GAMMA)

    def test_geometry_helper_stays_pure_for_tiny_synthetic_rings(self):
        # Slot-math unit tests drive ``_build_noncausal_indices`` with toy rings;
        # the capacity guard belongs to the forward, not this pure helper, so it
        # must not raise here.
        slots, _ = _BUILD_NONCAUSAL(
            _model(),
            torch.tensor([6]),
            torch.tensor([True]),
            torch.tensor([[1]]),
            4,
            4,
        )
        self.assertEqual(tuple(slots.shape), (_GAMMA, _TOP_INDEX_WIDTH))

    def test_topk_width_and_valid_prefix_cover_committed_plus_query(self):
        slots, length = _BUILD_NONCAUSAL(
            _model(),
            torch.tensor([200]),
            torch.tensor([True]),
            torch.tensor([[3, 4]]),
            _RING,
            _TOKENS_PER_BLOCK,
        )
        # With window=128, gamma=5 the physical width is one 128 page doubled.
        self.assertEqual(int(slots.shape[-1]), _TOP_INDEX_WIDTH)
        # topk_length is exactly committed + gamma; the rest is padding.
        self.assertEqual(int(length[0]), _WINDOW + _GAMMA)
        per_row_valid = int((slots[0] >= 0).sum())
        self.assertEqual(per_row_valid, _WINDOW + _GAMMA)

    def test_query_slots_never_collide_with_the_window_they_read(self):
        # 200: mid block, 254: query block straddles the block-0/1 boundary,
        # 300: window itself spans two physical blocks.
        for prefix in (200, 254, 300):
            with self.subTest(prefix=prefix):
                slots, _ = _BUILD_NONCAUSAL(
                    _model(),
                    torch.tensor([prefix]),
                    torch.tensor([True]),
                    torch.tensor([[3, 4]]),
                    _RING,
                    _TOKENS_PER_BLOCK,
                )
                row = slots[0].to(torch.long)
                committed = min(prefix, _WINDOW)
                context = row[:committed]
                query = row[committed : committed + _GAMMA]
                valid = row[row >= 0]
                # The whole read set is distinct, so no row aliases another.
                self.assertEqual(valid.numel(), valid.unique().numel())
                # And the freshly written query slots never overwrite a window slot.
                self.assertEqual(int(torch.isin(query, context).sum()), 0)

    def test_commit_writer_and_propose_reader_agree_on_context_slots(self):
        for prefix in (200, 254, 300):
            with self.subTest(prefix=prefix):
                committed = min(prefix, _WINDOW)
                start = prefix - committed
                context_positions = list(range(start, prefix))
                block_ids = [3 if pos // _TOKENS_PER_BLOCK == 0 else 4 for pos in context_positions]
                writer = _commit_slots(block_ids, context_positions, seq_len=prefix)
                # Every context slot the reader will touch must be a real write,
                # not a ring generation the writer deliberately skipped.
                self.assertTrue(bool((writer >= 0).all()), f"skipped context slot at prefix={prefix}")
                slots, _ = _BUILD_NONCAUSAL(
                    _model(),
                    torch.tensor([prefix]),
                    torch.tensor([True]),
                    torch.tensor([[3, 4]]),
                    _RING,
                    _TOKENS_PER_BLOCK,
                )
                reader_ctx = slots[0][:committed].to(torch.long)
                self.assertTrue(torch.equal(writer, reader_ctx), f"slot mismatch at prefix={prefix}")

    def test_inactive_request_reads_nothing(self):
        slots, length = _BUILD_NONCAUSAL(
            _model(),
            torch.tensor([200]),
            torch.tensor([False]),
            torch.tensor([[3, 4]]),
            _RING,
            _TOKENS_PER_BLOCK,
        )
        self.assertEqual(int(length[0]), 0)
        self.assertTrue(bool((slots < 0).all()))


class DsparkPdHandoffContractTest(unittest.TestCase):
    """Pin the draft-side SWA pool source and PD publication wiring by source."""

    def _method(self, name):
        tree = ast.parse(_SRC.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4DSparkModel"
        )
        method = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return ast.unparse(method)

    def test_commit_publishes_the_draft_swa_pool_per_layer_when_prefilling(self):
        body = self._method("commit_feature_rows")
        # The store op is the shared PD writer and may be None for decode calls.
        self.assertIn("create_write_cache_store_impl", body)
        self.assertIn("if write_cache_store_impl is not None:", body)
        # Publication is the draft's own SWA layer cache, per layer, inside the loop.
        self.assertIn("get_layer_cache(layer_idx, SWA_KV)", body)

    def test_block_table_comes_from_the_swa_group_and_is_batch_sliced(self):
        body = self._method("_swa_block_table")
        self.assertIn("build_block_tables_for_tags", body)
        self.assertIn("(SWA_KV,)", body)
        self.assertIn("table[:batch_size]", body)

    def test_propose_builds_noncausal_indices_before_attending(self):
        body = self._method("_forward_dspark_attention")
        # The ring-capacity guard runs where the real pool size is observable.
        self.assertIn("_require_swa_ring_capacity", body)
        # Query rows are written before the packed window+query read is built.
        self.assertIn("_build_noncausal_indices", body)
        self.assertLess(body.index("decode_write_swa_fp8("), body.index("_build_noncausal_indices"))


if __name__ == "__main__":
    unittest.main()
