import os
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils.dsa_indexing import build_dsa_indexer_kv_slot_mapping
from rtp_llm.utils.glm5_layer_repeat import (
    build_execution_layers,
    checkpoint_layer_config,
    configure_glm5_layer_repeat,
    expand_repeated_layer_weights,
)


def config():
    return SimpleNamespace(
        model_type="glm_5",
        num_layers=78,
        is_mtp=False,
        attn_config=SimpleNamespace(use_mla=True, is_sparse=True),
        moe_layer_index=list(range(3, 78)),
        index_topk_freq=4,
        index_skip_topk_offset=3,
        eplb_config=SimpleNamespace(enable_eplb=lambda: False),
    )


class Glm5LayerRepeatTest(unittest.TestCase):
    def expand(self, c, selection="middle27"):
        with patch.dict(os.environ, {"GLM5_REPEAT_MOE_LAYERS": selection}):
            configure_glm5_layer_repeat(c)
        return c

    def test_105_execution_layers_and_78_checkpoint_layers(self):
        c = self.expand(config())
        counts = Counter(c._glm5_layer_source_ids)
        self.assertEqual(c.num_layers, 105)
        self.assertEqual([i for i in range(78) if counts[i] == 2], list(range(27, 54)))
        self.assertEqual(c.moe_layer_index, list(range(3, 105)))
        self.assertEqual(c._glm5_layer_source_ids[:29], list(range(28)) + [27])
        self.assertEqual(c._glm5_layer_source_ids[-1], 77)
        source = checkpoint_layer_config(c)
        self.assertEqual(source.num_layers, 78)
        self.assertEqual(source.moe_layer_index, list(range(3, 78)))
        self.assertEqual(len(source.indexer_types), 78)
        source.embedding_size = 123
        self.assertEqual(c.embedding_size, 123)
        self.expand(c)
        self.assertEqual(c.num_layers, 105)

    def test_compact_indexer_has_distinct_slots_for_repeated_full_layers(self):
        c = self.expand(config())
        mapping = build_dsa_indexer_kv_slot_mapping(c, c.num_layers)
        self.assertEqual(len(mapping), 105)
        self.assertEqual(max(mapping) + 1, 27)
        for i in range(1, 105):
            if c._glm5_layer_source_ids[i] == c._glm5_layer_source_ids[i - 1]:
                if c.indexer_types[i] == "full":
                    self.assertNotEqual(mapping[i], mapping[i - 1])
                else:
                    self.assertEqual(mapping[i], mapping[i - 1])

    def test_loader_aliases_weights_and_initializes_each_source_once(self):
        c = self.expand(config())
        original = [{"weight": object()} for _ in range(78)]
        weights = SimpleNamespace(weights=original)
        expand_repeated_layer_weights(c, weights)
        self.assertEqual(len(weights.weights), 105)
        self.assertEqual(len({id(w) for w in weights.weights}), 78)
        created = []

        def create(i):
            created.append(i)
            return SimpleNamespace(logical_id=i, weights=weights.weights[i])

        def repeat(source, i):
            return SimpleNamespace(logical_id=i, weights=source.weights)

        layers = build_execution_layers(c, create, repeat)
        self.assertEqual(len(created), 78)
        for i, layer in enumerate(layers):
            self.assertEqual(layer.logical_id, i)
            self.assertIs(layer.weights, original[c._glm5_layer_source_ids[i]])
        with self.assertRaises(ValueError):
            expand_repeated_layer_weights(c, weights)

    def test_disabled_and_mtp_are_unchanged(self):
        c = config()
        self.expand(c, "")
        self.assertEqual(c.num_layers, 78)
        self.assertIs(checkpoint_layer_config(c), c)
        c.is_mtp = True
        c.num_layers = 1
        c.mtp_layer_offset = 78
        self.expand(c)
        self.assertEqual((c.num_layers, c.mtp_layer_offset), (1, 78))
        self.assertFalse(hasattr(c, "_glm5_layer_source_ids"))

    def test_invalid_selections_fail_before_config_mutation(self):
        for value in [
            "0,1,2",
            ",".join(["27"] * 27),
            ",".join(map(str, range(27))),
            "bad",
        ]:
            c = config()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.expand(c, value)
            self.assertEqual(c.num_layers, 78)
        for change in [dict(num_layers=4), dict(model_type="deepseek3")]:
            c = config()
            c.__dict__.update(change)
            with self.assertRaises(ValueError):
                self.expand(c)

    def test_unsupported_attention_and_eplb_fail_without_mutation(self):
        for mode in ("dense_attention", "non_sparse", "eplb"):
            c = config()
            if mode == "dense_attention":
                c.attn_config.use_mla = False
            elif mode == "non_sparse":
                c.attn_config.is_sparse = False
            else:
                c.eplb_config.enable_eplb = lambda: True
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.expand(c)
            self.assertEqual(c.num_layers, 78)
            self.assertFalse(hasattr(c, "_glm5_layer_source_ids"))

    def test_explicit_pattern_is_preserved_through_repeat_and_checkpoint_view(self):
        c = config()
        c.indexer_types = ["full" if i % 3 == 0 else "shared" for i in range(78)]
        pattern = c.indexer_types.copy()
        self.expand(c)
        self.assertEqual(
            c.indexer_types, [pattern[i] for i in c._glm5_layer_source_ids]
        )
        self.assertEqual(checkpoint_layer_config(c).indexer_types, pattern)

    def test_checkpoint_python_helpers_use_source_layer_count(self):
        class SizingConfig(SimpleNamespace):
            def layer_size(self):
                return self.num_layers * 10

            def total_size(self):
                return self.layer_size() + 1

        c = self.expand(SizingConfig(**vars(config())))
        source = checkpoint_layer_config(c)
        self.assertEqual(source.total_size(), 781)
        self.assertEqual(c.total_size(), 1051)
        source.num_layers = 77
        self.assertEqual(c.num_layers, 105)

    def test_invalid_execution_plan_and_checkpoint_weight_count(self):
        c = self.expand(config())
        c.num_layers = 104
        with self.assertRaisesRegex(ValueError, "105 layers"):
            self.expand(c)
        with self.assertRaisesRegex(ValueError, "execution plan"):
            build_execution_layers(c, lambda i: i, lambda s, i: i)
        with self.assertRaisesRegex(ValueError, "original checkpoint"):
            expand_repeated_layer_weights(c, SimpleNamespace(weights=[{}] * 77))

    def test_disabled_helpers_preserve_normal_model(self):
        c = config()
        weights = SimpleNamespace(weights=[{} for _ in range(78)])
        original = weights.weights
        expand_repeated_layer_weights(c, weights)
        self.assertIs(weights.weights, original)
        self.assertEqual(build_execution_layers(c, lambda i: i, None), list(range(78)))

    def test_explicit_noncontiguous_layers(self):
        ids = list(range(3, 57, 2))
        c = self.expand(config(), ",".join(map(str, ids)))
        counts = Counter(c._glm5_layer_source_ids)
        self.assertEqual([i for i in range(78) if counts[i] == 2], ids)


if __name__ == "__main__":
    unittest.main()
