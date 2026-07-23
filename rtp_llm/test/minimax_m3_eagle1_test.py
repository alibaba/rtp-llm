import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.minimax_m3_eagle1 import (
    MiniMaxM3Eagle1,
    _external_lm_head_path,
    _load_external_lm_head,
)
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeModel,
)
from rtp_llm.models_py.model_desc.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3Model,
    _expand_target_verify_rows,
    _target_verify_width,
    _update_target_verify_rope_kv_offset,
    _validate_target_verify_replay_shape,
)
from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model
from rtp_llm.models_py.modules.hybrid.msa_attention import (
    MSAAttention,
    _build_target_verify_token_metadata,
    _repeat_request_block_table_for_verify_tokens,
)


class EagleConfigTest(unittest.TestCase):
    def test_rejects_multi_layer_hass_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            config = {
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "vocab_size": 32,
            }
            with open(Path(tmpdir) / "config.json", "w") as writer:
                json.dump(config, writer)

            with self.assertRaisesRegex(ValueError, "exactly one draft layer"):
                MiniMaxM3Eagle1._create_config(tmpdir)


class EagleExternalLmHeadTest(unittest.TestCase):
    def test_loads_lm_head_from_bundle_assets_sibling(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = root / "draft_model"
            assets = root / "assets"
            ckpt.mkdir()
            assets.mkdir()
            expected = torch.randn(3, 4, dtype=torch.bfloat16)
            torch.save(expected, assets / "lm_head.pt")

            self.assertEqual(
                _external_lm_head_path(str(ckpt)), str(assets / "lm_head.pt")
            )
            actual = _load_external_lm_head([], ckpt_path=str(ckpt))
            torch.testing.assert_close(actual, expected)

    def test_rejects_missing_lm_head(self):
        with TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "draft_model"
            ckpt.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "external lm_head"):
                _load_external_lm_head([], ckpt_path=str(ckpt))


class EagleFcInputTest(unittest.TestCase):
    def test_hass_input_normalizes_embedding_and_hidden_before_projection(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value + 1,
            hidden_norm=lambda value: value * 2,
        )
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)

        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)

        torch.testing.assert_close(
            actual, torch.cat([embedding + 1, hidden * 2], dim=-1)
        )

    def test_hass_input_rejects_wrong_target_hidden_width(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value,
            hidden_norm=lambda value: value,
        )
        with self.assertRaisesRegex(RuntimeError, "HASS draft expected target hidden"):
            MiniMaxM3Eagle1Model._build_fc_input(
                draft, torch.randn(2, 4), torch.randn(2, 5)
            )


class DecoderAttentionHookTest(unittest.TestCase):
    def test_generic_decoder_keeps_causal_attention_call_contract(self):
        class FakeCausalAttention:
            def __init__(self):
                self.qkv_proj = object()
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["hidden_states"] + 1

        layer = object.__new__(GenericMoeDecoderLayer)
        torch.nn.Module.__init__(layer)
        attention = FakeCausalAttention()
        layer.self_attn = attention
        hidden_states = torch.randn(2, 4)
        fp8_states = torch.randn(2, 4)
        fp8_scale = torch.randn(2, 1)

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.CausalAttention",
            FakeCausalAttention,
        ):
            self.assertIs(layer._input_quant_projection(), attention.qkv_proj)
            actual, topk = layer._forward_attention(
                hidden_states,
                fmha_impl="fmha",
                kv_cache="kv_cache",
                prev_topk_indices=None,
                force_reuse_topk_indices=False,
                attn_inputs="unused",
                x_fp8=fp8_states,
                x_scale=fp8_scale,
            )

        torch.testing.assert_close(actual, hidden_states + 1)
        self.assertIsNone(topk)
        self.assertEqual(attention.kwargs["fmha_impl"], "fmha")
        self.assertEqual(attention.kwargs["kv_cache"], "kv_cache")
        self.assertIs(attention.kwargs["x_fp8"], fp8_states)
        self.assertIs(attention.kwargs["x_scale"], fp8_scale)

    def test_minimax_decoder_owns_msa_attention_call_contract(self):
        class FakeMSAAttention:
            def __init__(self):
                self.qkv_proj = object()
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["hidden_states"] + 2

        layer = object.__new__(MiniMaxM3DecoderLayer)
        torch.nn.Module.__init__(layer)
        attention = FakeMSAAttention()
        layer.self_attn = attention
        hidden_states = torch.randn(2, 4)
        fp8_states = torch.randn(2, 4)
        fp8_scale = torch.randn(2, 1)
        attention_inputs = object()

        with patch(
            "rtp_llm.models_py.model_desc.minimax_m3.MSAAttention",
            FakeMSAAttention,
        ):
            self.assertIs(layer._input_quant_projection(), attention.qkv_proj)
            actual, topk = layer._forward_attention(
                hidden_states,
                fmha_impl="unused",
                kv_cache="kv_cache",
                prev_topk_indices=None,
                force_reuse_topk_indices=False,
                attn_inputs=attention_inputs,
                x_fp8=fp8_states,
                x_scale=fp8_scale,
            )

        torch.testing.assert_close(actual, hidden_states + 2)
        self.assertIsNone(topk)
        self.assertIs(attention.kwargs["attn_inputs"], attention_inputs)
        self.assertEqual(attention.kwargs["kv_cache"], "kv_cache")
        self.assertIs(attention.kwargs["x_fp8"], fp8_states)
        self.assertIs(attention.kwargs["x_scale"], fp8_scale)
        self.assertNotIn("fmha_impl", attention.kwargs)


class TargetVerifyAttentionContractTest(unittest.TestCase):
    def test_updates_graph_owned_rope_kv_offset_in_place(self):
        captured_offset = torch.zeros((2, 1, 2, 3), dtype=torch.int32)
        converted_offset = torch.arange(12, dtype=torch.int32).reshape(2, 1, 2, 3)
        rope_params = SimpleNamespace(kv_cache_offset=captured_offset)
        block_table = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)

        with patch(
            "rtp_llm.models_py.model_desc.minimax_m3.convert_offset_to_block_array",
            return_value=converted_offset,
        ) as convert:
            _update_target_verify_rope_kv_offset(rope_params, block_table)

        convert.assert_called_once_with(block_table)
        self.assertIs(rope_params.kv_cache_offset, captured_offset)
        torch.testing.assert_close(captured_offset, converted_offset)

    def test_minimax_uses_shared_attention_contract_outside_target_verify(self):
        self.assertFalse(
            hasattr(MiniMaxM3Model, "prepare_target_verify_attention_inputs")
        )
        model = object.__new__(MiniMaxM3Model)
        torch.nn.Module.__init__(model)
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_target_verify=False)
        )
        with patch.object(
            GenericMoeModel, "prepare_fmha_impl", return_value="shared"
        ) as shared_prepare:
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(actual, "shared")
        shared_prepare.assert_called_once_with(inputs, True)

    def test_expands_request_metadata_for_target_verify_attention(self):
        prefix_lengths = torch.tensor([3, 7], dtype=torch.int32)
        block_table = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)

        sequence_lengths, token_block_table = _expand_target_verify_rows(
            prefix_lengths, block_table, verify_tokens=3
        )

        torch.testing.assert_close(
            sequence_lengths,
            torch.tensor([4, 5, 6, 8, 9, 10], dtype=torch.int32),
        )
        torch.testing.assert_close(
            token_block_table,
            torch.tensor(
                [
                    [11, 12],
                    [11, 12],
                    [11, 12],
                    [21, 22],
                    [21, 22],
                    [21, 22],
                ],
                dtype=torch.int32,
            ),
        )

    def test_masks_cuda_graph_padding_rows(self):
        sequence_lengths, _ = _expand_target_verify_rows(
            torch.tensor([3, 0], dtype=torch.int32),
            torch.tensor([[11, 12], [0, 0]], dtype=torch.int32),
            verify_tokens=2,
            valid_requests=torch.tensor([True, False]),
        )

        torch.testing.assert_close(
            sequence_lengths,
            torch.tensor([4, 5, 0, 0], dtype=torch.int32),
        )

    def test_derives_verify_width_from_flat_token_window(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(2, dtype=torch.int32), total_tokens=6
        )

        self.assertEqual(_target_verify_width(attn_inputs), 3)

    def test_derives_verify_width_from_cuda_graph_capture_placeholder(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            input_lengths=torch.full((2,), 4, dtype=torch.int32),
            total_tokens=0,
        )

        self.assertEqual(_target_verify_width(attn_inputs), 4)

    def test_rejects_variable_width_cuda_graph_capture_placeholder(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            input_lengths=torch.tensor([4, 3], dtype=torch.int32),
            total_tokens=0,
        )

        with self.assertRaisesRegex(RuntimeError, "one fixed width"):
            _target_verify_width(attn_inputs)

    def test_rejects_non_rectangular_verify_window(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(2, dtype=torch.int32), total_tokens=5
        )

        with self.assertRaisesRegex(RuntimeError, "divisible by request rows"):
            _target_verify_width(attn_inputs)

    def test_replay_keeps_capture_width_for_partially_filled_batch_bucket(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(4, dtype=torch.int32), total_tokens=12
        )

        _validate_target_verify_replay_shape(attn_inputs, verify_tokens=4)

    def test_replay_rejects_incomplete_request_window(self):
        attn_inputs = SimpleNamespace(
            prefix_lengths=torch.zeros(8, dtype=torch.int32), total_tokens=27
        )

        with self.assertRaisesRegex(RuntimeError, "incomplete request window"):
            _validate_target_verify_replay_shape(attn_inputs, verify_tokens=4)

    def test_minimax_target_verify_selects_model_local_impl(self):
        calls = []

        class FakeTargetVerifyImpl:
            def __init__(self, attn_configs, attn_inputs, parallelism_config):
                calls.append((attn_configs, attn_inputs, parallelism_config))

        model = object.__new__(MiniMaxM3Model)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(getAttentionConfigs=lambda tp_size: tp_size)
        model.parallelism_config = "parallelism"
        attn_inputs = SimpleNamespace(is_target_verify=True, is_cuda_graph=False)
        inputs = SimpleNamespace(attention_inputs=attn_inputs)

        with patch(
            "rtp_llm.models_py.model_desc.minimax_m3._target_verify_impl_class",
            return_value=FakeTargetVerifyImpl,
        ):
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertIsInstance(actual, FakeTargetVerifyImpl)
        self.assertEqual(calls, [(1, attn_inputs, "parallelism")])
        self.assertTrue(attn_inputs.is_cuda_graph)


class TargetVerifyBlockTableTest(unittest.TestCase):
    def test_expands_request_rows_to_verify_token_rows(self):
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        actual = _repeat_request_block_table_for_verify_tokens(
            table, batch_size=2, total_tokens=6
        )
        expected = torch.tensor(
            [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_divisible_token_rows(self):
        with self.assertRaisesRegex(RuntimeError, "batch \* verify_tokens"):
            _repeat_request_block_table_for_verify_tokens(
                torch.zeros((2, 3), dtype=torch.int32), batch_size=2, total_tokens=5
            )

    def test_rejects_wrong_block_rows_for_single_verify_token(self):
        with self.assertRaisesRegex(RuntimeError, "block table batch mismatch"):
            _repeat_request_block_table_for_verify_tokens(
                torch.zeros((1, 3), dtype=torch.int32), batch_size=2, total_tokens=2
            )

    def test_msa_selects_existing_grouped_kernel_table_locally(self):
        attention = object.__new__(MSAAttention)
        attention.layer_idx = 3
        attention.page_size = 128
        attention.physical_page_size = 128
        group0 = torch.tensor([[1, 2]], dtype=torch.int32)
        group1 = torch.tensor([[3, 4]], dtype=torch.int32)
        inputs = SimpleNamespace(
            kv_cache_layer_to_group=torch.tensor([0, 0, 0, 1]),
            kv_cache_kernel_block_id_device_by_group=[group0, group1],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=group0,
        )

        self.assertIs(attention._physical_block_table(inputs), group1)

    def test_msa_rejects_kernel_table_as_physical_table_for_different_page_sizes(self):
        attention = object.__new__(MSAAttention)
        attention.layer_idx = 0
        attention.page_size = 64
        attention.physical_page_size = 128
        inputs = SimpleNamespace(
            kv_cache_layer_to_group=torch.tensor([0]),
            kv_cache_kernel_block_id_device_by_group=[
                torch.tensor([[1, 2]], dtype=torch.int32)
            ],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=torch.tensor([[1, 2]], dtype=torch.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "physical page table"):
            attention._physical_block_table(inputs)


class TargetVerifyTokenMetadataTest(unittest.TestCase):
    def test_expands_request_positions_and_masks_cuda_graph_padding(self):
        positions, sequence_lengths, valid_tokens = _build_target_verify_token_metadata(
            prefix_lengths=torch.tensor([10, 20, 0], dtype=torch.int32),
            input_lengths=torch.tensor([3, 3, 0], dtype=torch.int32),
            total_tokens=9,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            positions,
            torch.tensor([10, 11, 12, 20, 21, 22, 0, 1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            sequence_lengths,
            torch.tensor([11, 12, 13, 21, 22, 23, 0, 0, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            valid_tokens,
            torch.tensor([True, True, True, True, True, True, False, False, False]),
        )

    def test_rejects_input_length_batch_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "input length batch mismatch"):
            _build_target_verify_token_metadata(
                prefix_lengths=torch.zeros(2, dtype=torch.int32),
                input_lengths=torch.ones(1, dtype=torch.int32),
                total_tokens=4,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
