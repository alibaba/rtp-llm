"""CPU planning coverage using the Python model and native binding interfaces."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttnImplFactory


class BertEncoderGraphMetadataTest(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace()
        self.metadata.request_capacity = 3
        self.metadata.token_capacity = 8
        self.metadata.max_sequence_length = None
        self.metadata.cu_seqlens_host = torch.empty(5, dtype=torch.int32)

    def _prepare(self, inputs, for_capture=False):
        return common.prepare_embedding_graph_lengths(
            inputs,
            self.metadata.token_capacity,
            self.metadata.request_capacity,
            self.metadata.max_sequence_length,
            self.metadata.cu_seqlens_host,
            for_capture,
        )

    @staticmethod
    def _inputs(lengths, prefix=None):
        return SimpleNamespace(
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(
                prefix or [0] * len(lengths), dtype=torch.int32
            ),
        )

    def test_layout_changes_keep_storage_and_isolate_padding(self):
        pointer = self.metadata.cu_seqlens_host.data_ptr()
        cases = [
            ([8, 0, 0], [0, 8, 8, 8, 8]),
            ([2, 3, 0], [0, 2, 5, 5, 8]),
            ([1, 1, 1], [0, 1, 2, 3, 8]),
            ([0, 0, 0], [0, 0, 0, 0, 8]),
            ([3, 0, 4], [0, 3, 3, 7, 8]),
        ]
        for lengths, expected in cases:
            with self.subTest(lengths=lengths):
                self._prepare(self._inputs(lengths))
                self.assertEqual(self.metadata.cu_seqlens_host.tolist(), expected)
                self.assertEqual(self.metadata.cu_seqlens_host.data_ptr(), pointer)

    def test_incompatible_profiles_are_rejected(self):
        for lengths, prefix in [
            ([9, 0, 0], None),
            ([-1, 2, 0], None),
            ([1, 2, 0], [0, 1, 0]),
            ([1, 2], None),
        ]:
            with self.subTest(lengths=lengths, prefix=prefix):
                with self.assertRaises(ValueError):
                    self._prepare(self._inputs(lengths, prefix))

    def test_32_item_batch_uses_aggregate_bucket(self):
        self.metadata.request_capacity = 32
        self.metadata.token_capacity = 6144
        self.metadata.max_sequence_length = 512
        self.metadata.cu_seqlens_host = torch.empty(45, dtype=torch.int32)
        # Same item text lengths as mock_request/bert/request.json, plus four
        # visual rows and a seven-token query per item.
        item_lengths = [
            182,
            192,
            149,
            114,
            192,
            161,
            165,
            140,
            150,
            136,
            136,
            167,
            141,
            138,
            153,
            151,
            192,
            178,
            164,
            188,
            177,
            156,
            171,
            151,
            149,
            136,
            116,
            174,
            180,
            142,
            185,
            124,
        ]
        lengths = [length + 4 + 7 for length in item_lengths]
        self._prepare(self._inputs(lengths))
        self.assertEqual(int(self.metadata.cu_seqlens_host[32]), 5402)
        self.assertEqual(int(self.metadata.cu_seqlens_host[-1]), 6144)
        self.assertEqual(self.metadata.cu_seqlens_host.numel(), 45)

    def test_single_sequence_limit_is_independent_of_total_capacity(self):
        self.metadata.token_capacity = 6144
        self.metadata.max_sequence_length = 512
        with self.assertRaisesRegex(ValueError, "position-table"):
            self._prepare(self._inputs([513, 0, 0]))
        # Aggregate capture profiles are split across legal request slots.
        self.metadata.request_capacity = 16
        self.metadata.cu_seqlens_host = torch.empty(29, dtype=torch.int32)
        inputs = self._inputs([6144] + [0] * 15)
        self._prepare(inputs, for_capture=True)
        self.assertEqual(int(self.metadata.cu_seqlens_host[-1]), 6144)
        self.assertTrue(bool((self.metadata.cu_seqlens_host.diff() <= 512).all()))

    def test_capture_fills_remaining_capacity_with_bounded_padding(self):
        self.metadata.token_capacity = 16
        self.metadata.max_sequence_length = 4
        self.metadata.cu_seqlens_host = torch.empty(8, dtype=torch.int32)
        self._prepare(self._inputs([16, 0, 0]), for_capture=True)
        self.assertEqual(
            self.metadata.cu_seqlens_host.tolist(), [0, 4, 8, 12, 16, 16, 16, 16]
        )


class EmbeddingGraphContractTest(unittest.TestCase):
    def test_backend_capability_checks_rope_and_backend_availability(self):
        from rtp_llm.ops import RopeStyle

        config = SimpleNamespace(
            use_mla=False,
            dtype=torch.bfloat16,
            head_num=4,
            kv_head_num=2,
            size_per_head=64,
            rope_config=SimpleNamespace(
                style=RopeStyle.No, dim=0, offset=0, scale=1.0
            ),
        )
        with patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(
                common.embedding_graph_attention_unsupported_reason(config, "flashinfer"), ""
            )
            config.rope_config.style = RopeStyle.Base
            self.assertIn(
                "RoPE",
                common.embedding_graph_attention_unsupported_reason(config, "flashinfer"),
            )
        config.rope_config.style = RopeStyle.No
        with patch("importlib.util.find_spec", return_value=None):
            self.assertIn(
                "not installed",
                common.embedding_graph_attention_unsupported_reason(config, "flashinfer"),
            )

    def test_bert_accepts_metadata_only_inputs_with_undefined_token_tensor(self):
        model = BertModel.__new__(BertModel)
        torch.nn.Module.__init__(model)
        inputs = SimpleNamespace(
            input_ids=None,
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[], mm_features_locs=None
            ),
            embedding_inputs=SimpleNamespace(text_tokens_mask=None),
        )
        self.assertTrue(model.validate_embedding_input_features(inputs))



    def test_default_adapter_rejects_explicit_positions_before_graph_planning(self):
        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        attention = PyAttentionInputs()
        inputs = SimpleNamespace(
            attention_inputs=attention,
            combo_position_ids=None,
            multimodal_inputs=SimpleNamespace(
                multimodal_features=[], mm_features_locs=None
            ),
            embedding_inputs=SimpleNamespace(text_tokens_mask=None),
        )
        self.assertTrue(model.validate_embedding_input_features(inputs))
        attention.combo_position_ids = torch.tensor([5, 6], dtype=torch.int32)
        self.assertFalse(model.validate_embedding_input_features(inputs))
        attention.combo_position_ids = torch.empty(0, dtype=torch.int32)
        self.assertTrue(model.validate_embedding_input_features(inputs))
        inputs.combo_position_ids = torch.tensor([5, 6], dtype=torch.int32)
        self.assertFalse(model.validate_embedding_input_features(inputs))

    def test_model_validation_rejects_bad_lengths_and_prefixes_before_planning(self):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        class TextModel(GptModelBase):
            def validate_embedding_input_features(self, inputs):
                return True

        model = TextModel.__new__(TextModel)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(max_seq_len=8)
        attention = PyAttentionInputs()
        attention.is_prefill, attention.is_target_verify = True, False
        attention.input_lengths = torch.tensor([3, 2], dtype=torch.int32)
        attention.prefix_lengths = torch.zeros(2, dtype=torch.int32)
        inputs = SimpleNamespace(attention_inputs=attention)
        self.assertTrue(model.validate_embedding_graph_inputs(inputs))
        self.assertTrue(model.validate_embedding_graph_inputs(inputs))
        for lengths, prefix in (
            ([9, 1], [0, 0]),
            ([-1, 1], [0, 0]),
            ([3, 2], [0]),
            ([3, 2], [1, 0]),
        ):
            attention.input_lengths = torch.tensor(lengths, dtype=torch.int32)
            attention.prefix_lengths = torch.tensor(prefix, dtype=torch.int32)
            self.assertFalse(model.validate_embedding_graph_inputs(inputs))

    def test_replay_prepares_each_tag_against_its_fixed_metadata(self):
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        groups = {tag: PyAttentionInputs() for tag in ("full", "aux")}
        attention = {tag: Mock() for tag in groups}
        captured = SimpleNamespace(attention_inputs=groups)
        model.prepare_embedding_graph_replay(attention, captured, None)
        for tag, impl in attention.items():
            impl.prepare_cuda_graph.assert_called_once_with(groups[tag])

    def test_missing_split_contract_is_unsupported(self):
        model = GptModelBase.__new__(GptModelBase)
        torch.nn.Module.__init__(model)
        self.assertIn("split", model.embedding_cuda_graph_unsupported_reason())
        with self.assertRaises(NotImplementedError):
            model.prepare_transformer_inputs(None)

    def test_bert_graph_reuses_forward_without_preparing_embeddings(self):
        model = BertModel.__new__(BertModel)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList()
        model.layer_num = 0
        hidden = torch.randn(3, 4)
        attention = SimpleNamespace()
        for inputs in (
            SimpleNamespace(),
            SimpleNamespace(input_hiddens=None),
            SimpleNamespace(input_hiddens=torch.empty(0)),
        ):
            with self.subTest(inputs=inputs), patch.object(
                model, "prepare_transformer_inputs", return_value=hidden
            ) as prepare:
                model.prepare_forward_inputs(inputs)
                eager = model.forward(inputs, attention).hidden_states
                prepare.assert_called_once_with(inputs)
                torch.testing.assert_close(eager, hidden)
        inputs = SimpleNamespace(input_hiddens=hidden)
        with patch.object(
            model,
            "prepare_transformer_inputs",
            side_effect=AssertionError("graph must reuse prepared embeddings"),
        ):
            replay = model.forward(inputs, attention).hidden_states
        torch.testing.assert_close(replay, eager)


    def test_factory_uses_registered_device_backend_for_embedding_graph(self):
        from rtp_llm.models_py.modules.factory.attention import attn_factory
        from rtp_llm.ops.compute_ops import PyAttentionInputs

        class DevicePrefillBackend:
            embedding_cuda_graph_unsupported_reason = Mock(return_value="")
            support_parallelism_config = Mock(return_value=True)

            def __init__(self, *args, **kwargs):
                self.arguments = kwargs

        attn = PyAttentionInputs()
        attn.is_prefill = True
        attn.padding_offset = torch.empty(8, dtype=torch.int32)
        config = SimpleNamespace(use_mla=False)
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", [DevicePrefillBackend]):
            self.assertEqual(
                AttnImplFactory.embedding_cuda_graph_unsupported_reason(config), ""
            )
            impl = attn_factory.get_fmha_impl(
                config,
                None,
                attn,
                max_seq_len=4,
                is_cuda_graph=True,
                cuda_graph_selection_mode="embedding_graph",
            )
            self.assertIsInstance(impl, DevicePrefillBackend)
            self.assertEqual(
                impl.arguments,
                dict(is_cuda_graph=True, token_capacity=8, max_seq_len=4),
            )
        with patch.object(attn_factory, "PREFILL_MHA_IMPS", []):
            self.assertIn(
                "no registered",
                AttnImplFactory.embedding_cuda_graph_unsupported_reason(config),
            )


if __name__ == "__main__":
    unittest.main()
