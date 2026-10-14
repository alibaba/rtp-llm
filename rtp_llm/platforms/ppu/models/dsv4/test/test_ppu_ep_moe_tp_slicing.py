"""TP token slicing for the PPU EP MoE DeepEP dispatch (source-extracted).

The helpers are extracted with ``ast`` so the test runs on hosts without the
PPU runtime; the collective import inside ``_gather_tp_slices`` is stubbed
through ``sys.modules`` and restored afterwards.
"""

import ast
import sys
import types
import unittest
from pathlib import Path

import torch

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "ppu_ep_moe.py"
_COLLECTIVE_MODULE = "rtp_llm.models_py.distributed.collective_torch"


def _load_helpers():
    tree = ast.parse(_SOURCE_PATH.read_text())
    wanted = {"_slice_tokens_for_tp", "_gather_tp_slices"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(_SOURCE_PATH), "exec"),
        namespace,
    )
    return namespace


class TpSlicingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_helpers()

    def _slices(self, tokens, ids, positions, mask, tp_size):
        results = []
        for rank in range(tp_size):
            results.append(
                self.helpers["_slice_tokens_for_tp"](
                    tokens, ids, positions, mask, tp_size, rank
                )
            )
        return results

    def test_slices_partition_the_replicated_batch_exactly_once(self):
        for total in (0, 1, 3, 8, 10, 17, 64):
            for tp_size in (1, 2, 4, 8):
                with self.subTest(total=total, tp_size=tp_size):
                    tokens = torch.arange(total * 3, dtype=torch.float32).view(
                        total, 3
                    )
                    ids = torch.arange(total)
                    positions = torch.arange(total)
                    mask = torch.ones(total, dtype=torch.bool)
                    results = self._slices(tokens, ids, positions, mask, tp_size)
                    self.assertEqual([r[4] for r in results], [total] * tp_size)
                    sizes = [r[0].size(0) for r in results]
                    self.assertEqual(sum(sizes), total)
                    self.assertLessEqual(max(sizes), (total + tp_size - 1) // tp_size)
                    rejoined = torch.cat([r[0] for r in results], dim=0)
                    self.assertTrue(torch.equal(rejoined, tokens))
                    for result in results:
                        self.assertEqual(result[1].numel(), result[0].size(0))
                        self.assertEqual(result[2].numel(), result[0].size(0))
                        self.assertEqual(result[3].numel(), result[0].size(0))
                        self.assertTrue(result[0].is_contiguous())

    def test_gather_reconstructs_the_replicated_batch(self):
        for total in (0, 1, 3, 8, 10, 17):
            for tp_size in (1, 2, 8):
                with self.subTest(total=total, tp_size=tp_size):
                    tokens = torch.randn(total, 5)
                    ids = torch.arange(total)
                    results = self._slices(
                        tokens, ids, None, None, tp_size
                    )
                    tp_tokens = (total + tp_size - 1) // tp_size
                    world = []
                    for rank in range(tp_size):
                        local = results[rank][0]
                        if local.size(0) < tp_tokens:
                            padding = torch.zeros(
                                (tp_tokens - local.size(0), 5),
                                dtype=local.dtype,
                            )
                            local = torch.cat((local, padding), dim=0)
                        world.append(local)

                    def fake_all_gather(tensor, group):
                        self.assertEqual(group, "TP")
                        return torch.cat(world, dim=0)

                    restored = self._gather_with_stub(
                        results[0][0], total, tp_size, fake_all_gather
                    )
                    self.assertEqual(restored.shape, tokens.shape)
                    self.assertTrue(torch.equal(restored, tokens))

    def _gather_with_stub(self, out, total, tp_size, fake_all_gather):
        stub = types.ModuleType(_COLLECTIVE_MODULE)
        stub.Group = type("Group", (), {"TP": "TP"})
        stub.all_gather = fake_all_gather
        parents = []
        prefix = ""
        for part in _COLLECTIVE_MODULE.split(".")[:-1]:
            prefix = f"{prefix}.{part}" if prefix else part
            module = sys.modules.get(prefix)
            if module is None:
                module = types.ModuleType(prefix)
                sys.modules[prefix] = module
                parents.append(prefix)
            else:
                parents.append(None)
        previous = sys.modules.get(_COLLECTIVE_MODULE)
        sys.modules[_COLLECTIVE_MODULE] = stub
        try:
            return self.helpers["_gather_tp_slices"](out, total, tp_size)
        finally:
            if previous is None:
                sys.modules.pop(_COLLECTIVE_MODULE, None)
            else:
                sys.modules[_COLLECTIVE_MODULE] = previous
            for name in parents:
                if name is not None:
                    sys.modules.pop(name, None)

    def test_forward_slices_before_the_capacity_chunk_loop(self):
        source = _SOURCE_PATH.read_text()
        self.assertIn("_slice_tokens_for_tp(", source)
        self.assertIn("_gather_tp_slices(out, original_tokens, self._tp_size)", source)
        loop = source.index("for start in range(")
        slicing = source.index("_slice_tokens_for_tp(")
        self.assertLess(slicing, loop)
        # The decode capacity guard still bounds the full replicated batch.
        self.assertIn("Decode or graph MoE input exceeds max_tokens_per_rank", source)


if __name__ == "__main__":
    unittest.main()
