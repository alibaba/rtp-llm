import unittest
from types import SimpleNamespace

from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer, dsa_layer_skips_topk


def _config(**kwargs):
    values = {"attn_config": SimpleNamespace(is_sparse=True)}
    values.update(kwargs)
    return SimpleNamespace(**values)


class DsaIndexingTest(unittest.TestCase):
    def test_glm52_indexer_types(self):
        config = _config(
            indexer_types=[
                "full",
                "full",
                "full",
                "shared",
                "shared",
                "shared",
                "full",
                "shared",
            ]
        )

        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertFalse(dsa_layer_skips_topk(config, 2))
        self.assertTrue(dsa_layer_skips_topk(config, 3))
        self.assertTrue(dsa_layer_skips_topk(config, 5))
        self.assertFalse(dsa_layer_skips_topk(config, 6))
        self.assertTrue(dsa_layer_skips_topk(config, 7))

        self.assertTrue(dsa_layer_has_indexer(config, 0))
        self.assertFalse(dsa_layer_has_indexer(config, 3))
        self.assertTrue(dsa_layer_has_indexer(config, 6))

    def test_offset_frequency_fallback(self):
        config = _config(index_topk_freq=4, index_skip_topk_offset=3)

        full_layers = {0, 1, 2, 6, 10}
        for layer_idx in range(12):
            self.assertEqual(
                dsa_layer_skips_topk(config, layer_idx), layer_idx not in full_layers
            )

    def test_mtp_layer_always_has_indexer(self):
        config = _config(
            is_mtp=True,
            indexer_types=["shared"],
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )

        self.assertFalse(dsa_layer_skips_topk(config, 0))
        self.assertTrue(dsa_layer_has_indexer(config, 0))

    def test_non_sparse_layer_has_no_indexer(self):
        config = _config(attn_config=SimpleNamespace(is_sparse=False))

        self.assertFalse(dsa_layer_has_indexer(config, 0))

    def test_invalid_indexer_type_fails(self):
        config = _config(indexer_types=["unknown"])

        with self.assertRaises(ValueError):
            dsa_layer_skips_topk(config, 0)


if __name__ == "__main__":
    unittest.main()
