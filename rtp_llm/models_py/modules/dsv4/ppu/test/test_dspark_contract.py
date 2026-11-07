"""CPU geometry checks without importing the native RTP runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[5]


def load_functions(path, names, scope, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope


class DSparkContractTest(unittest.TestCase):
    def test_noncausal_window_preserves_all_five_queries_across_ring_and_page_boundaries(self):
        scope = load_functions(
            ROOT / "models_py/model_desc/deepseek_v4_dspark_model.py",
            {"_global_pool_slots", "_build_noncausal_indices"}, {"torch": torch},
            "DeepSeekV4DSparkModel",
        )
        slots = scope["_global_pool_slots"]
        if isinstance(slots, staticmethod):
            slots = slots.__func__
        owner = SimpleNamespace(_gen_num_per_cycle=5, _v4_args=SimpleNamespace(window_size=128), _global_pool_slots=slots)
        prefixes = [0, 1, 127, 128, 133, 134, 255, 256, 511]
        tables = torch.arange(1, len(prefixes) * 3 + 1).reshape(-1, 3)
        active = torch.ones(len(prefixes), dtype=torch.bool)
        active[-1] = False
        indices, lengths = scope["_build_noncausal_indices"](
            owner, torch.tensor(prefixes), active, tables, 134, 256,
        )
        indices = indices.reshape(len(prefixes), 5, 256)
        for b, prefix in enumerate(prefixes):
            positions = list(range(max(0, prefix - 128), prefix + 5)) if active[b] else []
            expected = [int(tables[b, p // 256]) * 134 + p % 134 for p in positions]
            self.assertEqual(int(lengths[b]), len(expected))
            for q in range(5):
                self.assertEqual(indices[b, q, :len(expected)].tolist(), expected)
                self.assertTrue(bool((indices[b, q, len(expected):] == -1).all()))

    def test_dspark_contract_does_not_relax_normal_swa(self):
        scope = load_functions(
            ROOT / "models_py/modules/dsv4/ppu/attention/attention_contract.py",
            {"_validate_model1_cache_tensor", "_validate_rich_pool", "_validate_rich_pool_page",
             "_validate_lengths", "validate_dspark_pool", "validate_swa_pool"},
            {"torch": torch, "RICH_SWA_PAGE": 128, "RICH_SWA_INDEX_WIDTH": 128},
        )
        pool = torch.empty_strided((3, 134, 584), (78336, 584, 1), dtype=torch.uint8)
        for width in (192, 256):
            kwargs = dict(q=torch.zeros(10, 1, 16, 512, dtype=torch.bfloat16),
                          swa_pool_3d=pool, swa_topk_3d=torch.zeros(10, 1, width, dtype=torch.int32),
                          attn_sink=torch.zeros(16), topk_length=torch.full((10,), 133, dtype=torch.int32))
            scope["validate_dspark_pool"](**kwargs)
            with self.assertRaisesRegex(ValueError, "unsupported width"):
                scope["validate_swa_pool"](**kwargs)
            kwargs["topk_length"] = torch.zeros(2, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "shape"):
                scope["validate_dspark_pool"](**kwargs)


if __name__ == "__main__":
    unittest.main()
