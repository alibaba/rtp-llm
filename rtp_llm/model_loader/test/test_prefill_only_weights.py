"""Headless decision exports must never load or recreate a vocabulary head."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import W


class StopBeforeCheckpointLoad(Exception):
    pass


class PrefillOnlyWeightLoadingTest(unittest.TestCase):
    def before_loading(self, handler, *, pruning=False):
        module = (
            SimpleNamespace(get_handler=lambda: handler)
            if handler is not None
            else None
        )
        config = SimpleNamespace(
            model_type="qwen35_dense",
            enable_w4a16_sm120_dense_ffn=False,
            enable_output_vocab_pruning=pruning,
            has_lm_head=True,
        )
        fake = SimpleNamespace(
            model_config=config,
            hw_kernel_config=SimpleNamespace(enable_cuda_graph=False),
            _init_custom_module=lambda: module,
            create_model_loader=Mock(side_effect=StopBeforeCheckpointLoad),
        )
        return fake

    def test_prefill_signal_disables_head_before_loader_is_constructed(self):
        fake = self.before_loading(SimpleNamespace(prefill_only=True))
        with self.assertRaises(StopBeforeCheckpointLoad):
            BaseModel.load(fake)
        self.assertFalse(fake.model_config.has_lm_head)
        fake.create_model_loader.assert_called_once_with()

    def test_ordinary_generation_and_generation_readout_keep_lm_head(self):
        for handler in (None, SimpleNamespace(prefill_only=False), SimpleNamespace()):
            fake = self.before_loading(handler)
            with self.subTest(handler=handler), self.assertRaises(
                StopBeforeCheckpointLoad
            ):
                BaseModel.load(fake)
            self.assertTrue(fake.model_config.has_lm_head)

    def test_vocab_pruning_is_rejected_before_weight_loading(self):
        fake = self.before_loading(SimpleNamespace(prefill_only=True), pruning=True)
        with self.assertRaisesRegex(ValueError, "vocabulary pruning"):
            BaseModel.load(fake)
        fake.create_model_loader.assert_not_called()

    def loader(self, *, has_lm_head=False, task_type=TaskType.LANGUAGE_MODEL):
        loader = object.__new__(ModelLoader)
        loader.model_config = SimpleNamespace(
            has_lm_head=has_lm_head,
            vocab_size=8,
            embedding_size=0,
            normalize_lm_head_weight=False,
            logit_scale=1.0,
        )
        loader._task_type = task_type
        loader._global_weight_aliases = {}
        loader._is_attn_model = True
        loader._load_config = SimpleNamespace(database=object(), num_layers=0)
        loader._misc_weights_info = []
        loader._weights_info = SimpleNamespace(
            model_config=loader.model_config,
            create_dynamic_weights=lambda: [],
        )
        return loader

    def test_scratch_and_fast_collectors_never_access_missing_head_tensors(self):
        loader = self.loader()
        embedding = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        embedding_descriptor = SimpleNamespace(
            name=W.embedding,
            load=Mock(return_value={W.embedding: embedding}),
            get_tensor_names=Mock(
                return_value={"model.language_model.embed_tokens.weight"}
            ),
        )
        forbidden = [
            SimpleNamespace(
                name=name,
                load=Mock(
                    side_effect=AssertionError("missing LM head must not be loaded")
                ),
                get_tensor_names=Mock(
                    side_effect=AssertionError(
                        "missing LM head must not get a collector"
                    )
                ),
            )
            for name in (W.lm_head, W.lm_head_b)
        ]
        loader._model_weights_info = SimpleNamespace(
            weights=[embedding_descriptor, *forbidden], layer_weights=[]
        )
        loaded = list(loader.prepare_weights("cpu"))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0][:2], (None, W.embedding))
        self.assertIs(loaded[0][2], embedding)
        tensor_map, descriptors = loader._generate_weight_info()
        self.assertEqual(set(tensor_map), {"model.language_model.embed_tokens.weight"})
        self.assertEqual([item.weight.name for item in descriptors], [W.embedding])
        for descriptor in forbidden:
            descriptor.load.assert_not_called()
            descriptor.get_tensor_names.assert_not_called()

    def test_headless_language_model_does_not_recreate_tied_embedding_head(self):
        loader = self.loader()
        embedding = torch.randn(8, 4)
        for stale_head in (False, True):
            weights = ModelWeights(0, "cpu", torch.float32)
            weights.set_global_weight(W.embedding, embedding)
            if stale_head:
                weights.set_global_weight(W.lm_head, torch.ones_like(embedding))
                weights.set_global_weight(W.lm_head_b, torch.ones(8))
            loader._load_dynamic_weights(weights, "cpu")
            self.assertNotIn(W.lm_head, weights.global_weights)
            self.assertNotIn(W.lm_head_b, weights.global_weights)
            self.assertIs(weights.get_global_weight(W.embedding), embedding)
            self.assertEqual(loader.model_config.embedding_size, 8)

    def test_normal_generation_preserves_explicit_and_tied_head_behavior(self):
        loader = self.loader(has_lm_head=True)
        embedding, head = torch.randn(8, 4), torch.randn(8, 4)
        for explicit_head in (False, True):
            weights = ModelWeights(0, "cpu", torch.float32)
            weights.set_global_weight(W.embedding, embedding)
            if explicit_head:
                weights.set_global_weight(W.lm_head, head)
            loader._load_dynamic_weights(weights, "cpu")
            self.assertIs(
                weights.get_global_weight(W.lm_head),
                head if explicit_head else embedding,
            )
        self.assertFalse(loader._maybe_skip_weight(SimpleNamespace(name=W.lm_head)))
        self.assertFalse(loader._maybe_skip_weight(SimpleNamespace(name=W.embedding)))

    def test_existing_non_generation_and_alias_skips_are_preserved(self):
        loader = self.loader(has_lm_head=True, task_type=TaskType.DENSE_EMBEDDING)
        self.assertTrue(loader._maybe_skip_weight(SimpleNamespace(name=W.lm_head)))
        self.assertFalse(loader._maybe_skip_weight(SimpleNamespace(name=W.embedding)))
        loader._global_weight_aliases = {W.embedding: torch.zeros(8, 4)}
        self.assertTrue(loader._maybe_skip_weight(SimpleNamespace(name=W.embedding)))


if __name__ == "__main__":
    unittest.main()
