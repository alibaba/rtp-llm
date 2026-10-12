import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_indexing():
    path = Path(__file__).resolve().parents[1] / "dsa_indexing.py"
    spec = importlib.util.spec_from_file_location("dsa_indexing_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DsaIndexingTest(unittest.TestCase):
    def setUp(self):
        self.indexing = load_indexing()

    def config(self, **kwargs):
        return SimpleNamespace(attn_config=SimpleNamespace(is_sparse=True), **kwargs)

    def test_explicit_shared_layers_use_latest_full_slot(self):
        cfg = self.config(index_topk_pattern="FFSSFS")
        self.assertEqual(
            self.indexing.build_dsa_indexer_kv_slot_mapping(cfg, 6),
            [0, 1, 1, 1, 2, 2],
        )

    def test_frequency_keeps_initial_full_layers(self):
        cfg = self.config(index_topk_freq=3)
        self.assertEqual(
            self.indexing.build_dsa_indexer_kv_slot_mapping(cfg, 8),
            [0, 1, 1, 1, 2, 2, 2, 3],
        )

    def test_mtp_always_owns_its_indexer(self):
        cfg = self.config(index_topk_pattern="SSSS", is_mtp=True)
        self.assertEqual(
            self.indexing.build_dsa_indexer_kv_slot_mapping(cfg, 4),
            [0, 1, 2, 3],
        )

    def test_invalid_patterns_fail_before_cache_allocation(self):
        with self.assertRaises(ValueError):
            self.indexing.build_dsa_indexer_kv_slot_mapping(
                self.config(index_topk_pattern="SF"), 2
            )
        with self.assertRaises(ValueError):
            self.indexing.dsa_layer_has_indexer(
                self.config(indexer_types=["unexpected"]), 0
            )


if __name__ == "__main__":
    unittest.main()
