"""CPU checks for the DSpARK non-causal query-block index geometry.

``_build_noncausal_indices`` is pure tensor arithmetic, so it can be exercised
without the native runtime: the two helpers are lifted out of the model source
and bound to a stub carrying only the two fields they read.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "rtp_llm/models_py/model_desc/deepseek_v4_dspark_model.py"
WANTED = ("_global_pool_slots", "_build_noncausal_indices")
RING_SOURCE = ROOT / "rtp_llm/models_py/kernels/ppu/attention/ppu_fp4_indexer_cache.py"


def _strip(node):
    """Drop decorators and annotations so the helper runs standalone on CPU."""
    node.decorator_list = []
    node.returns = None
    for arg in list(node.args.args) + list(node.args.kwonlyargs):
        arg.annotation = None
    return node


def _load_helpers():
    tree = ast.parse(SOURCE.read_text())
    model = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4DSparkModel"
    )
    nodes = [_strip(n) for n in model.body
             if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    assert len(nodes) == len(WANTED), f"missing helpers: found {[n.name for n in nodes]}"
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def _draft_class(namespace):
    """Minimal stand-in for the draft model: only what the helpers read.

    The helpers go on the class (not the instance) so ``_build_noncausal_indices``
    still binds ``self`` while ``_global_pool_slots`` stays a static method.
    """

    class Draft:
        def __init__(self, gamma=5, window=128):
            self._gen_num_per_cycle = gamma
            self._v4_args = SimpleNamespace(window_size=window)

    Draft._build_noncausal_indices = namespace["_build_noncausal_indices"]
    Draft._global_pool_slots = staticmethod(namespace["_global_pool_slots"])
    return Draft


class DsparkIndexGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_helpers()
        cls.Draft = _draft_class(cls.ns)

    def draft(self, gamma=5, window=128):
        return self.Draft(gamma=gamma, window=window)

    # Fixed pool geometry: one 256-token block, block id 7.
    ENTRIES = 256
    TOKENS = 256
    BLOCK = 7
    TOPK = 256

    def build(self, prefix, active, block_table=None, gamma=5):
        table = block_table if block_table is not None else torch.tensor([[self.BLOCK]])
        return self.draft(gamma=gamma)._build_noncausal_indices(
            torch.tensor([prefix], dtype=torch.int32),
            torch.tensor(active, dtype=torch.bool),
            table,
            self.ENTRIES,
            self.TOKENS,
        )

    def slot(self, position):
        return self.BLOCK * self.ENTRIES + position % self.ENTRIES

    def assert_padding(self, slots_row, valid):
        self.assertEqual(int((slots_row[:valid] < 0).sum()), 0, "valid prefix has -1")
        self.assertEqual(
            int((slots_row[valid:] != -1).sum()), 0, "padding is not -1"
        )

    def test_short_context_sees_window_and_whole_block(self):
        slots, length = self.build(prefix=10, active=[True])

        self.assertEqual(tuple(slots.shape), (5, self.TOPK))
        self.assertEqual(int(length[0]), 15)
        expected = torch.tensor([self.slot(p) for p in range(15)], dtype=torch.int32)
        self.assertTrue(torch.equal(slots[0, :15], expected))
        self.assert_padding(slots[0], 15)

        # Non-causal: every query row of the block sees the same slot set.
        for row in range(1, 5):
            self.assertTrue(torch.equal(slots[row], slots[0]))

    def test_long_context_keeps_only_the_latest_window(self):
        slots, length = self.build(prefix=200, active=[True])

        # committed = 128 leaves the window starting at 72, plus the 5 query rows.
        self.assertEqual(int(length[0]), 133)
        self.assertEqual(int(slots[0, 0]), self.slot(72))
        self.assertEqual(int(slots[0, 127]), self.slot(199))
        self.assertEqual(int(slots[0, 128]), self.slot(200))
        self.assertEqual(int(slots[0, 132]), self.slot(204))
        self.assert_padding(slots[0], 133)

    def test_empty_context_holds_only_the_query_block(self):
        slots, length = self.build(prefix=0, active=[True])

        self.assertEqual(int(length[0]), 5)
        expected = torch.tensor([self.slot(p) for p in range(5)], dtype=torch.int32)
        self.assertTrue(torch.equal(slots[0, :5], expected))
        self.assert_padding(slots[0], 5)

    def test_inactive_request_is_fully_masked(self):
        slots, length = self.build(prefix=200, active=[False])

        self.assertEqual(int(length[0]), 0)
        self.assertEqual(int((slots != -1).sum()), 0)

    def test_pool_slots_reject_unallocated_and_out_of_range_blocks(self):
        # tokens_per_block == entries_per_block == 4; block 0 means "unallocated".
        table = torch.tensor([[0, 5]])
        positions = torch.tensor([0, 1, 4, 5, 8], dtype=torch.long)
        req_ids = torch.zeros_like(positions)

        slots = self.Draft()._global_pool_slots(table, req_ids, positions, 4, 4)

        self.assertEqual(
            slots.tolist(),
            [-1, -1, 5 * 4 + 0, 5 * 4 + 1, -1],
        )


    def test_fp4_state_ring_accepts_every_page_aligned_width(self):
        # A compliant ring is window (2 C4 pages) plus the verify rows rounded up
        # to a page: 12 for MTP2 (gamma=2), 16 for DSpARK (gamma=5).
        ok = _load_ring_predicate()
        for width in (8, 12, 16, 20):
            self.assertTrue(ok(width), f"ring {width} must be admitted")
        for width in (0, 4, 6, 10, 13, 14):
            self.assertFalse(ok(width), f"ring {width} must be rejected")


def _load_ring_predicate():
    """Lift ``_c4_ring_ok`` out of the Triton module so it runs on CPU."""
    tree = ast.parse(RING_SOURCE.read_text())
    names = {"_c4_ring_ok", "C4_RING_MIN_ENTRIES"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    nodes += [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) in names for t in n.targets)
    ]
    assert len(nodes) == 2, f"ring helpers moved: {[ast.dump(n)[:40] for n in nodes]}"
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RING_SOURCE), "exec"), namespace)
    return namespace["_c4_ring_ok"]


if __name__ == "__main__":
    unittest.main()
