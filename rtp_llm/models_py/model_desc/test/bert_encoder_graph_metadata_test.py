"""CPU planning coverage using the Python model and native binding interfaces."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
    CacheFreePrefillGraphImpl,
)


class BertEncoderGraphMetadataTest(unittest.TestCase):
    def setUp(self):
        cls = CacheFreePrefillGraphImpl
        self.impl = cls.__new__(cls)
        self.impl.request_capacity = 3
        self.impl.token_capacity = 8
        self.impl.max_sequence_length = None
        self.impl.heads = self.impl.kv_heads = 1
        self.impl.dim = 64
        self.impl.dtype = torch.bfloat16
        self.impl.is_causal = False
        self.impl.rope_cache = None
        self.impl.positions_host = None
        self.impl.positions_device = None
        self.impl.padding_slots = 1
        self.impl.cu_seqlens_host = torch.empty(5, dtype=torch.int32)
        self.impl.wrapper = Mock()

    @staticmethod
    def _inputs(lengths, prefix=None):
        return SimpleNamespace(
            input_lengths=torch.tensor(lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(
                prefix or [0] * len(lengths), dtype=torch.int32
            ),
        )

    def test_layout_changes_keep_storage_and_isolate_padding(self):
        pointer = self.impl.cu_seqlens_host.data_ptr()
        cases = [
            ([8, 0, 0], [0, 8, 8, 8, 8]),
            ([2, 3, 0], [0, 2, 5, 5, 8]),
            ([1, 1, 1], [0, 1, 2, 3, 8]),
            ([0, 0, 0], [0, 0, 0, 0, 8]),
            ([3, 0, 4], [0, 3, 3, 7, 8]),
        ]
        for lengths, expected in cases:
            with self.subTest(lengths=lengths):
                self.impl.prepare_cuda_graph(self._inputs(lengths))
                self.assertEqual(self.impl.cu_seqlens_host.tolist(), expected)
                self.assertEqual(self.impl.cu_seqlens_host.data_ptr(), pointer)
                kwargs = self.impl.wrapper.plan.call_args.kwargs
                self.assertFalse(kwargs["causal"])
                self.assertTrue(kwargs["disable_split_kv"])

    def test_incompatible_profiles_are_rejected(self):
        for lengths, prefix in [
            ([9, 0, 0], None),
            ([-1, 2, 0], None),
            ([1, 2, 0], [0, 1, 0]),
            ([1, 2], None),
        ]:
            with self.subTest(lengths=lengths, prefix=prefix):
                with self.assertRaises(ValueError):
                    self.impl.prepare_cuda_graph(self._inputs(lengths, prefix))

    def test_32_item_batch_uses_aggregate_bucket(self):
        self.impl.request_capacity = 32
        self.impl.token_capacity = 6144
        self.impl.max_sequence_length = 512
        self.impl.padding_slots = 12
        self.impl.cu_seqlens_host = torch.empty(45, dtype=torch.int32)
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
        self.impl.prepare_cuda_graph(self._inputs(lengths))
        self.assertEqual(int(self.impl.cu_seqlens_host[32]), 5402)
        self.assertEqual(int(self.impl.cu_seqlens_host[-1]), 6144)
        self.assertEqual(self.impl.cu_seqlens_host.numel(), 45)

    def test_single_sequence_limit_is_independent_of_total_capacity(self):
        self.impl.token_capacity = 6144
        self.impl.max_sequence_length = 512
        with self.assertRaisesRegex(ValueError, "position-table"):
            self.impl.prepare_cuda_graph(self._inputs([513, 0, 0]))
        # Aggregate capture profiles are split across legal request slots.
        self.impl.request_capacity, self.impl.padding_slots = 16, 12
        self.impl.cu_seqlens_host = torch.empty(29, dtype=torch.int32)
        inputs = self._inputs([6144] + [0] * 15)
        self.impl._prepare(inputs, for_capture=True)
        self.assertEqual(int(self.impl.cu_seqlens_host[-1]), 6144)
        self.assertTrue(bool((self.impl.cu_seqlens_host.diff() <= 512).all()))

    def test_capture_fills_remaining_capacity_with_bounded_padding(self):
        self.impl.token_capacity = 16
        self.impl.max_sequence_length = 4
        self.impl.padding_slots = 4
        self.impl.cu_seqlens_host = torch.empty(8, dtype=torch.int32)
        self.impl._prepare(self._inputs([16, 0, 0]), for_capture=True)
        self.assertEqual(
            self.impl.cu_seqlens_host.tolist(), [0, 4, 8, 12, 16, 16, 16, 16]
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
                style=RopeStyle.Base, dim=64, offset=0, scale=1.0
            ),
        )
        with patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(CacheFreePrefillGraphImpl.unsupported_reason(config), "")
            for scale in (0.0, 0.5, float("nan"), float("inf")):
                config.rope_config.scale = scale
                self.assertIn(
                    "scale", CacheFreePrefillGraphImpl.unsupported_reason(config)
                )
        config.rope_config.scale = 1.0
        with patch("importlib.util.find_spec", return_value=None):
            self.assertIn(
                "not installed", CacheFreePrefillGraphImpl.unsupported_reason(config)
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
        attention.combo_position_ids = None
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

    def test_rotary_positions_reset_per_sequence_and_preserve_unrotated_tail(self):
        impl = CacheFreePrefillGraphImpl.__new__(CacheFreePrefillGraphImpl)
        angles = torch.tensor([[0.0, 0.0], [0.3, 0.7], [0.6, 1.4]])
        impl.rope_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
        impl.positions_device = torch.tensor([0, 1, 0, 1, 2, 0])
        q, k = torch.randn(6, 4, 6), torch.randn(6, 2, 6)
        actual_q, actual_k = impl._apply_rotary(q, k)
        for actual, original in ((actual_q, q), (actual_k, k)):
            expected = original.clone()
            for token, position in enumerate(impl.positions_device.tolist()):
                for head in range(original.size(1)):
                    for pair in range(2):
                        left, right = (
                            original[token, head, pair],
                            original[token, head, pair + 2],
                        )
                        angle = angles[position, pair]
                        expected[token, head, pair] = (
                            left * angle.cos() - right * angle.sin()
                        )
                        expected[token, head, pair + 2] = (
                            right * angle.cos() + left * angle.sin()
                        )
            torch.testing.assert_close(actual, expected)

    def test_planning_honors_causal_configuration(self):
        impl = CacheFreePrefillGraphImpl.__new__(CacheFreePrefillGraphImpl)
        impl.request_capacity, impl.token_capacity = 2, 8
        impl.max_sequence_length = 8
        impl.padding_slots = 1
        impl.cu_seqlens_host = torch.empty(4, dtype=torch.int32)
        impl.positions_host = None
        impl.heads, impl.kv_heads, impl.dim, impl.dtype = 4, 2, 64, torch.float16
        impl.wrapper = Mock()
        attn = SimpleNamespace(
            input_lengths=torch.tensor([3, 2], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
        )
        for causal in (True, False):
            impl.is_causal = causal
            impl.prepare_cuda_graph(attn)
            self.assertEqual(impl.wrapper.plan.call_args.kwargs["causal"], causal)
            self.assertEqual(impl.cu_seqlens_host.tolist(), [0, 3, 5, 8])


if __name__ == "__main__":
    unittest.main()
