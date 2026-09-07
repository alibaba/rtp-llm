"""Focused wiring tests for multimodal BERT embedding replacement."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.bert import (
    BertModel,
    _validate_bert_uqi_runtime,
)
from rtp_llm.models_py.modules import MultimodalEmbeddingInjector
from rtp_llm.ops.compute_ops import PyMultimodalInputs


class _EmbeddingStub(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab_size, hidden_size))
        self.hidden_size = hidden_size
        self.last_input_ids = None

    def forward(self, input_ids, *args, **kwargs):
        self.last_input_ids = input_ids.clone()
        return torch.zeros(input_ids.numel(), self.hidden_size)


class BertMultimodalForwardTest(unittest.TestCase):
    def _build_model(self, hidden_size: int = 4) -> BertModel:
        model = BertModel.__new__(BertModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(bert_uqi_config=SimpleNamespace(enabled=False))
        model.embed_tokens = _EmbeddingStub(vocab_size=128, hidden_size=hidden_size)
        model.pre_decoder_layernorm = nn.Identity()
        model.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        model.layers = nn.ModuleList()
        model.layer_num = 0
        model.kv_cache = None
        model.prepare_fmha_impl = lambda inputs: SimpleNamespace()
        return model

    @staticmethod
    def _make_inputs(input_ids, features, locs):
        multimodal_inputs = PyMultimodalInputs()
        multimodal_inputs.multimodal_features = features
        multimodal_inputs.mm_features_locs = torch.tensor(locs, dtype=torch.int32)
        text_mask = torch.ones(len(input_ids), dtype=torch.int32)
        for feature, loc in zip(features, locs):
            text_mask[loc : loc + feature.size(0)] = 0
        return SimpleNamespace(
            input_ids=torch.tensor(input_ids, dtype=torch.int32),
            embedding_inputs=SimpleNamespace(text_tokens_mask=text_mask),
            bert_embedding_inputs=SimpleNamespace(
                combo_position_ids=torch.empty(0),
                position_encoding=torch.empty(0),
                combo_tokens_type_ids=torch.empty(0),
                token_type_embedding=torch.empty(0),
                input_embedding_scalar=1.0,
            ),
            multimodal_inputs=multimodal_inputs,
        )

    def test_cpu_locations_avoid_device_readback_and_update_per_request(self):
        model = self._build_model()
        feature = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        for loc in (1, 2):
            with self.subTest(loc=loc):
                inputs = self._make_inputs([101, 7, 8, 9, 102], [feature], [])
                # A meta tensor has no storage to download. Successful execution
                # therefore proves the host mirror is used, not the device one.
                inputs.multimodal_inputs.mm_features_locs = torch.empty(
                    1, dtype=torch.int32, device="meta"
                )
                inputs.multimodal_inputs.mm_features_locs_host = torch.tensor(
                    [loc], dtype=torch.int32
                )
                inputs.embedding_inputs.text_tokens_mask[loc : loc + 2] = 0
                output = model.forward(inputs).hidden_states
                torch.testing.assert_close(output[loc : loc + 2], feature)
                self.assertEqual(
                    model.embed_tokens.last_input_ids[loc : loc + 2].tolist(), [0, 0]
                )

    def test_host_locations_are_optional_for_existing_callers(self):
        inputs = PyMultimodalInputs()
        self.assertIsNone(inputs.mm_features_locs_host)

    def test_replaces_only_placeholder_spans_before_embedding(self):
        model = self._build_model()
        feature = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        inputs = self._make_inputs([101, 7, -1, 9999, 102], [feature], [2])

        output = model.forward(inputs).hidden_states

        self.assertEqual(
            model.embed_tokens.last_input_ids.tolist(), [101, 7, 0, 0, 102]
        )
        torch.testing.assert_close(output[2:4], feature)
        torch.testing.assert_close(output[[0, 1, 4]], torch.zeros(3, 4))

    def test_no_features_preserves_input_ids(self):
        model = self._build_model()
        inputs = self._make_inputs([101, 7, 102], [], [])

        model.forward(inputs)

        self.assertEqual(model.embed_tokens.last_input_ids.tolist(), [101, 7, 102])

    def test_masked_embedding_does_not_mutate_request_ids(self):
        model = self._build_model()
        inputs = self._make_inputs([101, -1, -2, 102], [torch.zeros(2, 4)], [1])
        original_ids = inputs.input_ids.clone()
        original_mask = inputs.embedding_inputs.text_tokens_mask.clone()
        model.forward(inputs)
        torch.testing.assert_close(inputs.input_ids, original_ids)
        torch.testing.assert_close(inputs.embedding_inputs.text_tokens_mask, original_mask)
        self.assertEqual(model.embed_tokens.last_input_ids.tolist(), [101, 0, 0, 102])

    def test_rejects_missing_or_incorrectly_sized_mask(self):
        model = self._build_model()
        for mask in (None, torch.ones(3, dtype=torch.int32)):
            inputs = self._make_inputs([101, -1, -2, 102], [torch.zeros(2, 4)], [1])
            inputs.embedding_inputs.text_tokens_mask = mask
            with self.assertRaisesRegex(ValueError, "text mask must match"):
                model.forward(inputs)

    def test_rejects_invalid_feature_metadata_before_embedding(self):
        model = self._build_model()
        feature = torch.zeros(2, 4)

        with self.assertRaisesRegex(ValueError, "counts must match"):
            model.forward(self._make_inputs([101, -1, -1, 102], [feature], []))
        with self.assertRaisesRegex(IndexError, "outside"):
            model.forward(self._make_inputs([101, -1, -1, 102], [feature], [3]))


class BertUqiRuntimeValidationTest(unittest.TestCase):
    @staticmethod
    def _config(enabled: bool):
        return SimpleNamespace(
            bert_uqi_config=SimpleNamespace(enabled=enabled),
        )

    def test_rejects_layer_micro_batching(self):
        resources = SimpleNamespace(enable_layer_micro_batch=1)
        with self.assertRaisesRegex(ValueError, "enable_layer_micro_batch=0"):
            BertModel(
                self._config(enabled=True),
                parallelism_config=None,
                weights=None,
                max_generate_batch_size=1,
                device_resource_config=resources,
            )

    def test_allows_supported_runtime_modes(self):
        enabled = self._config(enabled=True)
        disabled = self._config(enabled=False)
        _validate_bert_uqi_runtime(enabled, None)
        _validate_bert_uqi_runtime(enabled, SimpleNamespace(enable_layer_micro_batch=0))
        _validate_bert_uqi_runtime(
            disabled, SimpleNamespace(enable_layer_micro_batch=1)
        )


if __name__ == "__main__":
    unittest.main()
