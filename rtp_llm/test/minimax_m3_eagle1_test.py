import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.minimax_m3 import _TARGET_LM_HEAD_BY_DEVICE, MiniMaxM3
from rtp_llm.models.minimax_m3_eagle1 import (
    MiniMaxM3Eagle1,
    MiniMaxM3Eagle1WeightInfo,
    MiniMaxM3Eagle1WeightNames,
)
from rtp_llm.models.qwen_v2 import QWenV2
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
    _prepare_target_verify_addressing,
    _repeat_request_block_table_for_verify_tokens,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.topk_sparse import (
    _merge_topk_attn_out,
    flash_decode_with_gqa_share_sparse_paged,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse import (
    minimax_paged_sparse_decode,
)
from rtp_llm.ops import RopeConfig, RopeStyle, get_rope_cache_once
from rtp_llm.utils.model_weight import W


class EagleConfigTest(unittest.TestCase):
    def test_draft_prefill_cuda_graph_uses_compact_token_capacity(self):
        self.assertFalse(
            getattr(
                MiniMaxM3Eagle1Model,
                "cuda_graph_prefill_requires_full_token_capacity",
                False,
            )
        )

    def test_loads_yarn_rope_scaling(self):
        with TemporaryDirectory() as tmpdir:
            config = {
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": 16.0,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0,
                },
            }
            with open(Path(tmpdir) / "config.json", "w") as writer:
                json.dump(config, writer)

            actual = MiniMaxM3Eagle1._create_config(tmpdir)

            self.assertEqual(actual.attn_config.rope_config.style, RopeStyle.Yarn)
            self.assertEqual(actual.attn_config.rope_config.scale, 16.0)
            self.assertEqual(actual.attn_config.rope_config.max_pos, 8192)
            self.assertEqual(actual.attn_config.rope_config.factor1, 1.0)
            self.assertEqual(actual.attn_config.rope_config.factor2, 32.0)
            self.assertGreater(actual.attn_config.rope_config.mscale, 1.0)
            self.assertFalse(actual.enable_fp32_lm_head)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_target_and_draft_rope_caches_are_isolated(self):
        target_rope = RopeConfig()
        target_rope.style = RopeStyle.Base
        target_rope.dim = 64
        target_rope.base = 5000003

        draft_rope = RopeConfig()
        draft_rope.style = RopeStyle.Yarn
        draft_rope.dim = 128
        draft_rope.base = 5000003
        draft_rope.scale = 16.0
        draft_rope.max_pos = 8192
        draft_rope.factor1 = 1.0
        draft_rope.factor2 = 32.0
        draft_rope.mscale = 1.277258872

        target_cache = get_rope_cache_once(
            target_rope, 131072, is_cuda=True, interleave=False
        )
        draft_cache = get_rope_cache_once(
            draft_rope, 131072, is_cuda=True, interleave=False
        )
        draft_cache_again = get_rope_cache_once(
            draft_rope, 131072, is_cuda=True, interleave=False
        )

        self.assertEqual(tuple(target_cache.data.shape), (131072, 64))
        self.assertEqual(tuple(draft_cache.data.shape), (131072, 128))
        self.assertNotEqual(target_cache.data.data_ptr(), draft_cache.data.data_ptr())
        self.assertEqual(draft_cache.data.data_ptr(), draft_cache_again.data.data_ptr())

        position = 80000
        frequency_index = 10
        correction = lambda rotations: (
            draft_rope.dim
            * math.log(draft_rope.max_pos / (rotations * 2.0 * math.pi))
            / (2.0 * math.log(draft_rope.base))
        )
        low = max(0, math.floor(correction(draft_rope.factor2)))
        high = min(
            draft_rope.dim - 1,
            math.ceil(correction(draft_rope.factor1)),
        )
        ramp = min(max((frequency_index - low) / (high - low), 0.0), 1.0)
        mask = (1.0 - ramp) * draft_rope.extrapolation_factor
        base_frequency = draft_rope.base ** (2.0 * frequency_index / draft_rope.dim)
        inverse_frequency = (1.0 / (draft_rope.scale * base_frequency)) * (
            1.0 - mask
        ) + (1.0 / base_frequency) * mask
        angle = position * inverse_frequency
        expected_cos = math.cos(angle) * draft_rope.mscale
        expected_sin = math.sin(angle) * draft_rope.mscale

        self.assertAlmostEqual(
            draft_cache.data[position, frequency_index].item(),
            expected_cos,
            delta=1e-3,
        )
        self.assertAlmostEqual(
            draft_cache.data[position, draft_rope.dim // 2 + frequency_index].item(),
            expected_sin,
            delta=1e-3,
        )

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


class EagleTargetLmHeadTest(unittest.TestCase):
    def setUp(self):
        _TARGET_LM_HEAD_BY_DEVICE.clear()

    def tearDown(self):
        _TARGET_LM_HEAD_BY_DEVICE.clear()

    def test_draft_checkpoint_does_not_load_lm_head(self):
        weight_info = object.__new__(MiniMaxM3Eagle1WeightInfo)
        weight_info._names = MiniMaxM3Eagle1WeightNames
        weight_info._hidden_size = 8
        weight_info._size_per_head = 4
        weight_info._head_num = 2
        weight_info._head_num_kv = 1
        weight_info._align_size = 0

        actual = weight_info._get_weight_info()

        self.assertNotIn(W.lm_head, [weight.name for weight in actual.weights])

    def test_draft_reuses_loaded_target_lm_head_tensor(self):
        # TP shards the target vocabulary dimension, while the draft embedding
        # remains full-vocabulary. The output head must retain the target shard.
        target_lm_head = torch.randn(2, 8)
        target_weights = ModelWeights(1, "cpu", target_lm_head.dtype)
        target_weights.set_global_weight(W.lm_head, target_lm_head)
        target = object.__new__(MiniMaxM3)
        target.weight = target_weights

        draft_embedding = torch.randn(4, 8)
        draft_weights = ModelWeights(1, "cpu", draft_embedding.dtype)
        draft_weights.set_global_weight(W.embedding, draft_embedding)
        draft_weights.set_global_weight(W.lm_head, draft_embedding)
        draft = object.__new__(MiniMaxM3Eagle1)
        draft.weight = draft_weights

        with patch.object(DeepSeekV2, "_load"):
            target._load("cpu")
        with patch.object(QWenV2, "_load"):
            draft._load("cpu")

        self.assertIs(draft.weight.get_global_weight(W.lm_head), target_lm_head)
        self.assertEqual(
            draft.weight.get_global_weight(W.lm_head).data_ptr(),
            target_lm_head.data_ptr(),
        )
        self.assertIs(draft.weight.get_global_weight(W.embedding), draft_embedding)


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
        parallelism = SimpleNamespace(get_attn_tp_size=lambda: 1)
        model.parallelism_config = parallelism
        attn_inputs = SimpleNamespace(is_target_verify=True, is_cuda_graph=False)
        inputs = SimpleNamespace(attention_inputs=attn_inputs)

        with patch(
            "rtp_llm.models_py.model_desc.minimax_m3._target_verify_impl_class",
            return_value=FakeTargetVerifyImpl,
        ):
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertIsInstance(actual, FakeTargetVerifyImpl)
        self.assertEqual(calls, [(1, attn_inputs, parallelism)])
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


class PagedDecodeAddressingCacheTest(unittest.TestCase):
    def setUp(self):
        self._previous_cache = MSAAttention._paged_decode_shared_meta
        MSAAttention._paged_decode_shared_meta = None

    def tearDown(self):
        MSAAttention._paged_decode_shared_meta = self._previous_cache

    def test_reuses_within_forward_and_rebuilds_on_layer_rollback(self):
        first_layer = object.__new__(MSAAttention)
        first_layer.layer_idx = 2
        first_layer.page_size = first_layer.physical_page_size = 128
        later_layer = object.__new__(MSAAttention)
        later_layer.layer_idx = 5
        later_layer.page_size = later_layer.physical_page_size = 128
        block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        inputs = SimpleNamespace(
            sequence_lengths=torch.tensor([10, 20], dtype=torch.int32),
            kv_cache_layer_to_group=None,
            kv_cache_kernel_block_id_device_by_group=[block_table],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=block_table,
        )

        first = first_layer._paged_decode_addressing(inputs, torch.device("cpu"))
        later = later_layer._paged_decode_addressing(inputs, torch.device("cpu"))
        self.assertTrue(all(lhs is rhs for lhs, rhs in zip(first, later)))

        inputs.sequence_lengths.add_(1)
        next_block_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id_device_by_group[0] = next_block_table
        inputs.kv_cache_kernel_block_id_device = next_block_table
        next_forward = first_layer._paged_decode_addressing(inputs, torch.device("cpu"))
        self.assertTrue(all(lhs is not rhs for lhs, rhs in zip(first, next_forward)))
        torch.testing.assert_close(
            next_forward[0], torch.tensor([12, 22], dtype=torch.int64)
        )
        torch.testing.assert_close(
            next_forward[1], torch.tensor([12, 22], dtype=torch.int32)
        )
        torch.testing.assert_close(
            next_forward[2], torch.tensor([11, 21], dtype=torch.int32)
        )

    def test_does_not_reuse_across_attention_input_owners(self):
        first_layer = object.__new__(MSAAttention)
        first_layer.layer_idx = 2
        first_layer.page_size = first_layer.physical_page_size = 128
        later_layer = object.__new__(MSAAttention)
        later_layer.layer_idx = 5
        later_layer.page_size = later_layer.physical_page_size = 128
        first_table = torch.tensor([[1, 2]], dtype=torch.int32)
        second_table = torch.tensor([[3, 4]], dtype=torch.int32)
        first_inputs = SimpleNamespace(
            sequence_lengths=torch.tensor([10], dtype=torch.int32),
            kv_cache_layer_to_group=None,
            kv_cache_kernel_block_id_device_by_group=[first_table],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=first_table,
        )
        second_inputs = SimpleNamespace(
            sequence_lengths=torch.tensor([20], dtype=torch.int32),
            kv_cache_layer_to_group=None,
            kv_cache_kernel_block_id_device_by_group=[second_table],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=second_table,
        )

        first = first_layer._paged_decode_addressing(first_inputs, torch.device("cpu"))
        second = later_layer._paged_decode_addressing(
            second_inputs, torch.device("cpu")
        )

        self.assertTrue(all(lhs is not rhs for lhs, rhs in zip(first, second)))
        torch.testing.assert_close(second[0], torch.tensor([21], dtype=torch.int64))
        self.assertIs(second[3], second_table)


class TargetVerifyTokenMetadataTest(unittest.TestCase):
    def setUp(self):
        self._previous_cache = MSAAttention._target_verify_shared_meta
        MSAAttention._target_verify_shared_meta = None

    def tearDown(self):
        MSAAttention._target_verify_shared_meta = self._previous_cache

    @staticmethod
    def _attention(layer_idx):
        attention = object.__new__(MSAAttention)
        attention.layer_idx = layer_idx
        attention.page_size = 128
        attention.physical_page_size = 128
        return attention

    @staticmethod
    def _inputs(block_table):
        return SimpleNamespace(
            prefix_lengths=torch.tensor([10, 20], dtype=torch.int32),
            input_lengths=torch.tensor([3, 3], dtype=torch.int32),
            kv_cache_layer_to_group=None,
            kv_cache_kernel_block_id_device_by_group=[block_table],
            kv_cache_block_id_device=None,
            kv_cache_kernel_block_id_device=block_table,
        )

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

    def test_reuses_addressing_across_increasing_sparse_layers(self):
        first_layer = self._attention(2)
        later_layer = self._attention(5)
        inputs = self._inputs(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

        first = first_layer._target_verify_addressing(
            inputs, total_tokens=6, device=torch.device("cpu")
        )
        later = later_layer._target_verify_addressing(
            inputs, total_tokens=6, device=torch.device("cpu")
        )

        self.assertTrue(all(lhs is rhs for lhs, rhs in zip(first, later)))

    def test_rebuilds_addressing_for_a_different_forward_owner(self):
        draft_layer = self._attention(2)
        target_layer = self._attention(5)
        draft_inputs = self._inputs(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
        target_inputs = self._inputs(torch.tensor([[5, 6], [7, 8]], dtype=torch.int32))
        target_inputs.prefix_lengths.add_(1)

        draft = draft_layer._target_verify_addressing(
            draft_inputs, total_tokens=6, device=torch.device("cpu")
        )
        target = target_layer._target_verify_addressing(
            target_inputs, total_tokens=6, device=torch.device("cpu")
        )

        self.assertTrue(all(lhs is not rhs for lhs, rhs in zip(draft, target)))
        torch.testing.assert_close(
            target[0], torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            target[2], torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32)
        )

    def test_fused_cuda_addressing_is_explicit(self):
        attention = self._attention(2)
        inputs = self._inputs(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

        with patch(
            "rtp_llm.models_py.modules.hybrid.msa_attention._prepare_target_verify_addressing",
            wraps=_prepare_target_verify_addressing,
        ) as prepare:
            attention._target_verify_addressing(
                inputs,
                total_tokens=6,
                device=torch.device("cpu"),
                use_fused_cuda=True,
            )

        self.assertTrue(prepare.call_args.kwargs["use_fused_cuda"])

        MSAAttention._target_verify_shared_meta = None
        with patch(
            "rtp_llm.models_py.modules.hybrid.msa_attention._prepare_target_verify_addressing",
            wraps=_prepare_target_verify_addressing,
        ) as prepare:
            attention._target_verify_addressing(
                inputs,
                total_tokens=6,
                device=torch.device("cpu"),
                use_fused_cuda=False,
            )

        self.assertFalse(prepare.call_args.kwargs["use_fused_cuda"])

    def test_rebuilds_addressing_when_layer_index_rolls_back(self):
        first_layer = self._attention(2)
        later_layer = self._attention(5)
        inputs = self._inputs(torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))

        first = first_layer._target_verify_addressing(
            inputs, total_tokens=6, device=torch.device("cpu")
        )
        later_layer._target_verify_addressing(
            inputs, total_tokens=6, device=torch.device("cpu")
        )
        inputs.prefix_lengths.add_(1)
        inputs.input_lengths[1] = 0
        next_block_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32)
        inputs.kv_cache_kernel_block_id_device_by_group[0] = next_block_table
        inputs.kv_cache_kernel_block_id_device = next_block_table
        next_forward = first_layer._target_verify_addressing(
            inputs, total_tokens=6, device=torch.device("cpu")
        )

        self.assertTrue(all(lhs is not rhs for lhs, rhs in zip(first, next_forward)))
        torch.testing.assert_close(
            next_forward[0],
            torch.tensor([[5, 6], [7, 8]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            next_forward[1],
            torch.tensor(
                [[5, 6], [5, 6], [5, 6], [7, 8], [7, 8], [7, 8]],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            next_forward[2],
            torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32),
        )
        torch.testing.assert_close(
            next_forward[3],
            torch.tensor([12, 13, 14, 0, 0, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            next_forward[4],
            torch.tensor([True, True, True, False, False, False]),
        )


class SparseDecodeRoutingTest(unittest.TestCase):
    @patch(
        "rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse."
        "flash_decode_with_gqa_share_sparse_paged"
    )
    @patch(
        "rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse."
        "flash_decode_with_topk_idx_paged"
    )
    def test_target_verify_uses_four_topk_chunks(self, mock_score, mock_sparse):
        for decode_query_len in (1, 2, 3, 4):
            with self.subTest(decode_query_len=decode_query_len):
                token_rows = 2 * decode_query_len
                q = torch.empty(token_rows, 1, 1, dtype=torch.bfloat16)
                topk_idx = torch.zeros(1, token_rows, 16, dtype=torch.int32)
                mock_score.return_value = (torch.empty_like(q), topk_idx)
                mock_sparse.return_value = q

                minimax_paged_sparse_decode(
                    q=q,
                    sink=None,
                    idx_q=q,
                    seq_lens=torch.ones(token_rows, dtype=torch.int32),
                    max_seqlen=128,
                    block_size_k=128,
                    topk=16,
                    init_blocks=1,
                    local_blocks=1,
                    paged_main_k=torch.empty(1, 1, 128, 1, dtype=torch.bfloat16),
                    paged_main_v=torch.empty(1, 1, 128, 1, dtype=torch.bfloat16),
                    phys_block_table=torch.zeros(token_rows, 1, dtype=torch.int32),
                    paged_idx_k=torch.empty(1, 128, 1, dtype=torch.bfloat16),
                    disable_index_value=True,
                    decode_query_len=decode_query_len,
                )

                sparse_kwargs = mock_sparse.call_args.kwargs
                self.assertEqual(
                    sparse_kwargs["num_topk_chunks"],
                    4 if decode_query_len > 1 else None,
                )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class SparseDecodeMergeTest(unittest.TestCase):
    def test_all_empty_chunks_merge_to_finite_zeros(self):
        o_partial = torch.randn(4, 3, 8, 128, dtype=torch.bfloat16, device="cuda")
        lse_partial = torch.full(
            (4, 3, 8), float("-inf"), dtype=torch.float32, device="cuda"
        )

        actual = _merge_topk_attn_out(o_partial, lse_partial)

        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, torch.zeros_like(actual))

    def test_general_paged_decode_handles_target_verify_shapes(self):
        torch.manual_seed(7)
        num_q_heads = 64
        num_kv_heads = 4
        head_dim = 128
        page_size = 128
        num_blocks = 4
        k_bf16 = torch.randn(
            num_blocks,
            num_kv_heads,
            page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        v_bf16 = torch.randn_like(k_bf16)

        for token_rows in (4, 6, 8):
            q = torch.randn(
                token_rows,
                num_q_heads,
                head_dim,
                dtype=torch.bfloat16,
                device="cuda",
            )
            block_table = torch.arange(
                num_blocks, dtype=torch.int32, device="cuda"
            ).repeat(token_rows, 1)
            seq_lens = torch.tensor(
                [0, 1, 127, 128, 129, 257, 511, 512][:token_rows],
                dtype=torch.int32,
                device="cuda",
            )
            topk_idx = (
                (torch.arange(16, dtype=torch.int32, device="cuda") % num_blocks)
                .view(1, 1, 16)
                .repeat(num_kv_heads, token_rows, 1)
            )
            topk_idx[:, :, 13:] = -1
            for kv_dtype in (torch.bfloat16, torch.float8_e4m3fn):
                k_paged = k_bf16.to(kv_dtype)
                v_paged = v_bf16.to(kv_dtype)
                for sink in (
                    None,
                    torch.randn(
                        num_q_heads,
                        head_dim,
                        dtype=torch.bfloat16,
                        device="cuda",
                    ),
                ):
                    common = dict(
                        q=q,
                        sink=sink,
                        k_paged=k_paged,
                        v_paged=v_paged,
                        block_table=block_table,
                        seq_lens=seq_lens,
                        block_size=page_size,
                        topk_idx=topk_idx,
                        num_topk_chunks=4,
                    )
                    actual = flash_decode_with_gqa_share_sparse_paged(**common)
                    self.assertTrue(torch.isfinite(actual).all().item())

                empty_topk = {
                    **common,
                    "sink": None,
                    "topk_idx": torch.full_like(topk_idx, -1),
                }
                actual_empty = flash_decode_with_gqa_share_sparse_paged(**empty_topk)
                self.assertTrue(torch.isfinite(actual_empty).all().item())
                torch.testing.assert_close(actual_empty, torch.zeros_like(actual_empty))

                non_divisible_topk = {
                    **common,
                    "sink": None,
                    "topk_idx": topk_idx[:, :, :15],
                }
                actual = flash_decode_with_gqa_share_sparse_paged(**non_divisible_topk)
                self.assertTrue(torch.isfinite(actual).all().item())


if __name__ == "__main__":
    unittest.main()
