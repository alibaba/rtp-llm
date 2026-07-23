import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models.minimax_m3_eagle3 import (
    MiniMaxM3Eagle3,
    MiniMaxM3Eagle3WeightInfo,
    MiniMaxM3Eagle3WeightNames,
    _merge_qkv_weight,
)
from rtp_llm.models_py.model_desc.minimax_m3 import _MiniMaxM3ModelMixin
from rtp_llm.models_py.model_desc.minimax_m3_eagle3 import (
    MiniMaxM3Eagle3DecoderLayer,
    MiniMaxM3Eagle3Model,
)
from rtp_llm.ops import SpeculativeType


def _draft_config(**overrides):
    config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "attention_bias": False,
        "draft_vocab_size": 32,
        "fc_norm": True,
        "head_dim": 128,
        "hidden_size": 8,
        "intermediate_size": 16,
        "norm_output": True,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "torch_dtype": "bfloat16",
        "vocab_size": 32,
    }
    config.update(overrides)
    return config


class Eagle3ConfigTest(unittest.TestCase):
    def test_parses_checkpoint_contract(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_draft_config()))

            config = MiniMaxM3Eagle3._create_config(tmpdir)

        self.assertEqual(config.model_type, "minimax_m3_eagle3")
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.hc_mult, 1)
        self.assertEqual(config.vocab_size, 32)
        self.assertEqual(config.config_dtype, "bfloat16")
        self.assertFalse(config.enable_fp32_lm_head)

    def test_rejects_unsupported_checkpoint_variant(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(
                json.dumps(_draft_config(norm_output=False))
            )
            with self.assertRaisesRegex(ValueError, "norm_output=true"):
                MiniMaxM3Eagle3._create_config(tmpdir)

    def test_rejects_wrong_head_dim(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(
                json.dumps(_draft_config(head_dim=64))
            )
            with self.assertRaisesRegex(ValueError, "head_dim=128"):
                MiniMaxM3Eagle3._create_config(tmpdir)

    def test_architecture_maps_only_to_minimax_eagle3(self):
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures("LlamaForCausalLMEagle3"),
            "minimax_m3_eagle3",
        )


class Eagle3HiddenContractTest(unittest.TestCase):
    def test_rejects_legacy_eagle_mode(self):
        sp_config = SimpleNamespace(type=SpeculativeType.EAGLE)
        target = SimpleNamespace(num_layers=60, hc_mult=1)
        draft = SimpleNamespace(hc_mult=1)

        with self.assertRaisesRegex(ValueError, "SP_TYPE=eagle3"):
            MiniMaxM3Eagle3.configure_speculative_model(sp_config, target, draft)

    def test_configures_default_target_layers(self):
        sp_config = SimpleNamespace(type=SpeculativeType.EAGLE3)
        target = SimpleNamespace(num_layers=60, hc_mult=1)
        draft = SimpleNamespace(hc_mult=1)

        MiniMaxM3Eagle3.configure_speculative_model(sp_config, target, draft)

        self.assertEqual(
            target._minimax_m3_eagle3_aux_hidden_state_layer_ids,
            (2, 30, 57),
        )
        self.assertEqual(target.hc_mult, 3)
        self.assertEqual(draft.hc_mult, 1)

    def test_uses_distinct_target_layers_for_shallow_debug_model(self):
        sp_config = SimpleNamespace(type=SpeculativeType.EAGLE3)
        target = SimpleNamespace(num_layers=5, hc_mult=1)
        draft = SimpleNamespace(hc_mult=1)

        MiniMaxM3Eagle3.configure_speculative_model(sp_config, target, draft)

        self.assertEqual(
            target._minimax_m3_eagle3_aux_hidden_state_layer_ids,
            (1, 3, 5),
        )

    def test_rejects_ambiguous_layer_order(self):
        sp_config = SimpleNamespace(type=SpeculativeType.EAGLE3)
        target = SimpleNamespace(num_layers=60, hc_mult=1)
        draft = SimpleNamespace(
            hc_mult=1,
            _minimax_m3_eagle3_aux_hidden_state_layer_ids=(30, 2, 57),
        )
        with self.assertRaisesRegex(ValueError, "invalid MiniMax-M3 EAGLE3"):
            MiniMaxM3Eagle3.configure_speculative_model(sp_config, target, draft)

    def test_target_hidden_getter_slices_latest_rows(self):
        model = SimpleNamespace(
            _mtp_target_hidden_states=torch.randn(5, 12),
        )
        actual = _MiniMaxM3ModelMixin.get_mtp_target_hidden_states(model, 3)
        self.assertEqual(tuple(actual.shape), (3, 12))

    def test_target_model_captures_configured_layer_outputs(self):
        hidden = torch.randn(2, 4)
        model = SimpleNamespace(
            _mtp_target_hidden_layer_ids=(0, 1, 2),
            _mtp_target_hidden_layer_slots={0: 0, 1: 1, 2: 2},
            _mtp_target_hidden_states=None,
        )

        capture = _MiniMaxM3ModelMixin._begin_mtp_target_hidden_capture(model, hidden)
        layer_1 = torch.randn(2, 4)
        residual_1 = torch.randn(2, 4)
        layer_2 = torch.randn(2, 4)
        residual_2 = torch.randn(2, 4)
        _MiniMaxM3ModelMixin._capture_mtp_target_hidden(
            model, capture, 1, layer_1, residual_1
        )
        _MiniMaxM3ModelMixin._capture_mtp_target_hidden(
            model, capture, 2, layer_2, residual_2
        )
        _MiniMaxM3ModelMixin._finish_mtp_target_hidden_capture(model, capture)

        torch.testing.assert_close(capture[:, :4], hidden)
        torch.testing.assert_close(capture[:, 4:8], layer_1 + residual_1)
        torch.testing.assert_close(capture[:, 8:], layer_2 + residual_2)
        self.assertIs(model._mtp_target_hidden_states, capture)


class Eagle3WeightTest(unittest.TestCase):
    def test_checkpoint_manifest_requires_all_weights(self):
        keys = MiniMaxM3Eagle3WeightNames.required()
        MiniMaxM3Eagle3WeightInfo._process_meta(SimpleNamespace(), [], keys)
        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            MiniMaxM3Eagle3WeightInfo._process_meta(
                SimpleNamespace(), [], keys - {"lm_head.weight"}
            )

    def test_checkpoint_manifest_rejects_unexpected_weights(self):
        keys = MiniMaxM3Eagle3WeightNames.required()
        with self.assertRaisesRegex(ValueError, "unexpected weights: extra.weight"):
            MiniMaxM3Eagle3WeightInfo._process_meta(
                SimpleNamespace(), [], keys | {"extra.weight"}
            )

    def test_qkv_merge_preserves_2h_input_width(self):
        q = torch.randn(4, 6)
        k = torch.randn(4, 6)
        v = torch.randn(4, 6)

        merged = _merge_qkv_weight([q, k, v])

        self.assertEqual(tuple(merged.shape), (6, 12))
        torch.testing.assert_close(merged[:, :4], q.T)
        torch.testing.assert_close(merged[:, 4:8], k.T)
        torch.testing.assert_close(merged[:, 8:], v.T)


class Eagle3ForwardContractTest(unittest.TestCase):
    def test_first_layer_builds_2h_attention_input(self):
        layer = SimpleNamespace(
            embedding_norm=lambda value: value + 1,
            hidden_norm=lambda value: value * 2,
        )
        embeds = torch.randn(2, 4)
        hidden = torch.randn(2, 4)

        actual = MiniMaxM3Eagle3DecoderLayer._build_attention_input(
            layer, embeds, hidden
        )

        torch.testing.assert_close(
            actual,
            torch.cat([embeds + 1, hidden * 2], dim=-1),
        )

    def test_autoregressive_draft_step_reuses_h_hidden(self):
        hidden = torch.randn(2, 4)
        model = SimpleNamespace(
            hidden_size=4,
            num_aux_hidden_states=3,
            _combine_aux_hidden_states=lambda value: value[:, :4],
        )

        actual = MiniMaxM3Eagle3Model._prepare_hidden_states(model, hidden)

        self.assertIs(actual, hidden)

    def test_target_step_projects_3h_hidden_once(self):
        hidden = torch.randn(2, 12)
        projected = torch.randn(2, 4)
        calls = []
        model = SimpleNamespace(
            hidden_size=4,
            num_aux_hidden_states=3,
            _combine_aux_hidden_states=lambda value: calls.append(value) or projected,
        )

        actual = MiniMaxM3Eagle3Model._prepare_hidden_states(model, hidden)

        self.assertIs(actual, projected)
        self.assertEqual(calls, [hidden])

    def test_target_hidden_chunks_are_contiguous_before_norm(self):
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        def require_contiguous(value):
            self.assertTrue(value.is_contiguous())
            return value

        model = SimpleNamespace(
            hidden_size=4,
            num_aux_hidden_states=3,
            fc_norms=[require_contiguous] * 3,
            fc=lambda value: value,
        )

        actual = MiniMaxM3Eagle3Model._combine_aux_hidden_states(model, hidden)

        torch.testing.assert_close(actual, hidden)


if __name__ == "__main__":
    unittest.main()
