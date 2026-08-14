import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.model_factory_register import ModelDict, get_lazy_model_module_path
from rtp_llm.models.minimax_m3_mtp import MiniMaxM3MTP, MiniMaxM3MTPWeight
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.generic_moe_mtp import GenericMoeMTPModel
from rtp_llm.models_py.model_desc.minimax_m3 import MiniMaxM3DecoderLayer
from rtp_llm.models_py.model_desc.minimax_m3_mtp import (
    MiniMaxM3MTPDecoderLayer,
    MiniMaxM3MTPModel,
    _MiniMaxM3MTPRefreshContext,
)
from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention
from rtp_llm.ops import KvCacheDataType, SpeculativeType
from rtp_llm.utils.model_weight import W


def _config(**overrides):
    text_config = {
        "hidden_size": 8,
        "head_dim": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 16,
        "num_hidden_layers": 2,
        "num_mtp_modules": 1,
        "intermediate_size": 4,
        "dense_intermediate_size": 8,
        "shared_intermediate_size": 4,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_layer_freq": [0, 1],
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_index_dim": 4,
            "sparse_num_index_heads": 1,
            "sparse_topk_blocks": 2,
            "sparse_block_size": 4,
            "sparse_attention_freq": [0, 1],
            "sparse_disable_index_value": [0, 1],
            "sparse_init_block": 1,
            "sparse_local_block": 1,
        },
    }
    text_config.update(overrides)
    return {
        "architectures": ["MiniMaxM3MTP"],
        "model_type": "minimax_m3_mtp",
        "text_config": text_config,
    }


def _weight_info():
    weight = object.__new__(MiniMaxM3MTPWeight)
    weight._load_raw_mxfp8_idx = False
    weight._num_layers = 1
    weight._hidden_size = 8
    weight._size_per_head = 4
    weight._head_num = 2
    weight._head_num_kv = 1
    weight._use_qk_norm = True
    weight._align_size = 0
    weight._is_gated_activation = True
    weight.moe_layer_index_ = [0]
    weight.expert_num_ = 2
    weight.prefix = "language_model."
    weight._mtp_root = "language_model.model.mtp.layers.0."
    weight._mtp_eh_proj_is_quantized = False
    weight._sparse_layer_set = {0}
    weight.has_e_score_correction_bias = True
    return weight


def _checkpoint_names(model_weight_info):
    names = set()
    modules = list(model_weight_info.weights)
    for layer in model_weight_info.layer_weights:
        modules.extend(layer)
    for module in modules:
        for component in module.get_components():
            for checkpoint_weight in getattr(component, "weights", []) or []:
                names.add(checkpoint_weight.name)
    return names


class MiniMaxM3MTPConfigTest(unittest.TestCase):
    def test_forces_one_sparse_moe_module(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_config()))
            config = MiniMaxM3MTP._create_config(tmpdir)

        self.assertEqual(config.model_type, "minimax_m3_mtp")
        self.assertEqual(config.num_layers, 1)
        self.assertEqual(config.moe_layer_index, [0])
        self.assertEqual(config.msa_sparse_config["sparse_layer_ids"], [0])
        self.assertEqual(config.msa_sparse_config["disable_value_layer_ids"], [0])
        self.assertEqual(config.attn_config.indexer_head_dim, 4)
        self.assertTrue(config.is_mtp)
        self.assertEqual(config.physical_mtp_module_num, 1)
        self.assertFalse(config.index_share_for_mtp_iteration)

    def test_rejects_multiple_physical_modules(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(
                json.dumps(_config(num_mtp_modules=2))
            )
            with self.assertRaisesRegex(ValueError, "num_mtp_modules=1"):
                MiniMaxM3MTP._create_config(tmpdir)

    def test_registration_is_explicit(self):
        self.assertEqual(
            get_lazy_model_module_path("minimax_m3_mtp"),
            "rtp_llm.models.minimax_m3_mtp",
        )
        self.assertEqual(
            ModelDict.get_ft_model_type_by_hf_architectures("MiniMaxM3MTP"),
            "minimax_m3_mtp",
        )

    def test_supports_bf16_and_fp8_draft_kv_cache(self):
        with TemporaryDirectory() as tmpdir:
            Path(tmpdir, "config.json").write_text(json.dumps(_config()))
            config = MiniMaxM3MTP._create_config(tmpdir)

        config.attn_config.kv_cache_dtype = KvCacheDataType.FP8
        MiniMaxM3MTP._validate_kv_cache_dtype(config)

        config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        MiniMaxM3MTP._validate_kv_cache_dtype(config)

        config.attn_config.kv_cache_dtype = KvCacheDataType.INT8
        with self.assertRaisesRegex(ValueError, "supports BF16 or FP8"):
            MiniMaxM3MTP._validate_kv_cache_dtype(config)


class MiniMaxM3MTPPrepareTest(unittest.TestCase):
    def test_refresh_bypasses_generic_dense_fmha_prepare(self):
        model = object.__new__(MiniMaxM3MTPModel)
        model._draft_prefill_runtime = False

        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_mtp_draft_prefill=True)
        )
        with patch.object(
            GenericMoeMTPModel,
            "prepare_fmha_impl",
            side_effect=AssertionError("dense FMHA prepare must not run"),
        ):
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=False)

        self.assertIsInstance(actual, _MiniMaxM3MTPRefreshContext)
        self.assertIsNone(actual.fmha_params)

    def test_regular_prefill_keeps_generic_fmha_prepare(self):
        model = object.__new__(MiniMaxM3MTPModel)
        model._draft_prefill_runtime = False
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_mtp_draft_prefill=False)
        )
        with patch.object(
            GenericMoeMTPModel,
            "prepare_fmha_impl",
            return_value="fmha",
        ) as prepare:
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertEqual(actual, "fmha")
        prepare.assert_called_once_with(inputs, True)

    def test_cuda_graph_clone_identity_selects_refresh_context(self):
        model = object.__new__(MiniMaxM3MTPModel)
        model._draft_prefill_runtime = True
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_mtp_draft_prefill=False)
        )
        with patch.object(
            GenericMoeMTPModel,
            "prepare_fmha_impl",
            side_effect=AssertionError("dense FMHA prepare must not run"),
        ):
            actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

        self.assertIsInstance(actual, _MiniMaxM3MTPRefreshContext)
        self.assertTrue(inputs.attention_inputs.is_mtp_draft_prefill)
        self.assertEqual(model._mtp_iteration_step(inputs), 0)


class MiniMaxM3MTPAttentionTest(unittest.TestCase):
    @staticmethod
    def _attention():
        attention = object.__new__(MSAAttention)
        torch.nn.Module.__init__(attention)
        attention._cuda_graph_max_seq_len = 8192
        attention.page_size = 128
        return attention

    def test_initial_cp_prefill_delegates_without_model_specific_cache_mutation(self):
        layer = object.__new__(MiniMaxM3MTPDecoderLayer)
        inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=object(),
            is_mtp_draft_prefill=False,
        )
        sentinel = {"owner": inputs, "layer_idx": 0}

        def forward(*_args, **_kwargs):
            MSAAttention._cp_shared_meta = sentinel
            return "hidden", None

        with patch.object(MSAAttention, "_cp_shared_meta", None), patch.object(
            MiniMaxM3DecoderLayer, "_forward_attention", side_effect=forward
        ):
            actual = layer._forward_attention(
                torch.empty(1, 8), None, None, None, False, inputs
            )

            self.assertIs(MSAAttention._cp_shared_meta, sentinel)

        self.assertEqual(actual, ("hidden", None))

    def test_raw_mxfp8_fused_projection_is_used_with_fp8_draft_kv(self):
        attention = self._attention()
        attention.head_num = 2
        attention.head_dim = 4
        attention.idx_head_dim = 4
        attention.block_size = 4
        attention.topk_blocks = 2
        attention.init_blocks = 1
        attention.local_blocks = 1
        attention.score_type = "dot"
        attention.disable_index_value = False
        attention.tp_size = 1
        attention.o_proj = nn.Identity()

        block_table = torch.zeros((1, 1), dtype=torch.int32)
        paged_kv = torch.empty((1, 2, 1, 128, 4), dtype=torch.float8_e4m3fn)
        # The idx_K side region is byte-backed and reinterpreted as BF16.
        kv_cache = SimpleNamespace(
            kv_cache_base=paged_kv,
            kv_scale_base=torch.empty((1, 1024), dtype=torch.uint8),
        )
        q = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
        idx_q = torch.zeros((1, 1, 4), dtype=torch.bfloat16)
        o = torch.ones((1, 2, 4), dtype=torch.bfloat16)

        with patch.object(
            attention,
            "_paged_decode_addressing",
            return_value=(
                torch.tensor([1], dtype=torch.int64),
                torch.tensor([1], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                block_table,
            ),
        ), patch.object(
            attention, "_should_use_mxfp8_fused_qkv_idx_decode", return_value=True
        ), patch.object(
            attention, "_paged_kv_base_view", return_value=paged_kv
        ), patch.object(
            attention, "_decode_project_fused_qkv_idx", return_value=(q, idx_q)
        ) as fused_project, patch.object(
            attention, "_paged_decode_max_kv", return_value=1
        ), patch(
            "rtp_llm.models_py.triton_kernels.sparse_msa.minimax_sparse.minimax_paged_sparse_decode",
            return_value=(torch.empty(0), o),
        ):
            actual = attention._forward_paged_decode(
                torch.zeros((1, 8), dtype=torch.bfloat16),
                SimpleNamespace(),
                kv_cache,
                x_fp8=torch.empty((1, 1), dtype=torch.uint8),
                x_scale=torch.ones((1, 1), dtype=torch.float32),
            )

        fused_project.assert_called_once()
        torch.testing.assert_close(actual, o.reshape(1, -1))

    def test_regular_eager_decode_keeps_exact_live_max(self):
        attention = self._attention()
        block_table = torch.zeros((2, 64), dtype=torch.int32)
        inputs = SimpleNamespace(mtp_iteration_step=-1)

        with patch.object(
            MSAAttention, "_cuda_graph_forward_active", return_value=False
        ):
            actual = attention._paged_decode_max_kv(
                inputs, torch.tensor([257, 511], dtype=torch.int32), block_table
            )

        self.assertEqual(actual, 511)

    def test_decode_side_mtp_prefill_uses_explicit_paged_continuation(self):
        attention = self._attention()
        inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=None,
            input_lengths=torch.empty(2, dtype=torch.int32),
        )
        hidden_states = torch.empty(10, 8)

        with patch.object(
            attention, "_forward_target_verify", return_value="paged"
        ) as forward:
            actual = attention.forward_paged_continuation(
                hidden_states, inputs, object()
            )
        self.assertEqual(actual, "paged")
        self.assertTrue(forward.call_args.kwargs["use_fused_addressing"])
        self.assertTrue(forward.call_args.kwargs["use_paged_capacity_bound"])

    def test_cp_refresh_preserves_existing_cp_prefill_fallback(self):
        attention = self._attention()
        inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=object(),
            input_lengths=torch.empty(1, dtype=torch.int32),
        )

        with patch.object(attention, "forward", return_value="cp") as forward:
            actual = attention.forward_paged_continuation(
                torch.empty(5, 8), inputs, object()
            )
        self.assertEqual(actual, "cp")
        forward.assert_called_once()

    def test_invalid_refresh_width_is_rejected(self):
        attention = self._attention()
        inputs = SimpleNamespace(
            is_prefill=True,
            context_parallel_info=None,
            input_lengths=torch.empty(2, dtype=torch.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "invalid recurrent MTP"):
            attention.forward_paged_continuation(torch.empty(5, 8), inputs, object())


class MiniMaxM3MTPWeightTest(unittest.TestCase):
    def test_bundled_checkpoint_names_map_only_to_mtp_module_zero(self):
        names = _checkpoint_names(_weight_info()._get_weight_info())

        self.assertIn("language_model.model.embed_tokens.weight", names)
        self.assertIn("language_model.lm_head.weight", names)
        self.assertIn("language_model.model.mtp.layers.0.eh_proj.weight", names)
        self.assertIn(
            "language_model.model.mtp.layers.0.transformer_layer.self_attn.q_proj.weight",
            names,
        )
        self.assertIn(
            "language_model.model.mtp.layers.0.transformer_layer.self_attn.index_q_proj.weight",
            names,
        )
        self.assertIn(
            "language_model.model.mtp.layers.0.transformer_layer.block_sparse_moe.experts.{expert_id}.w1.weight",
            names,
        )
        self.assertFalse(any("model.layers.{i}" in name for name in names))

    def test_bf16_eh_projection_uses_runtime_in_out_layout(self):
        weight_info = _weight_info()._get_weight_info()
        eh_proj = next(
            weight
            for weight in weight_info.weights
            if weight.name == W.multi_tokens_predict_eh_proj
        )
        checkpoint_weight = torch.randn(8, 16)

        loaded_weight = eh_proj.process_fun([checkpoint_weight])

        self.assertEqual(tuple(loaded_weight.shape), (16, 8))
        torch.testing.assert_close(loaded_weight, checkpoint_weight.T)
        self.assertTrue(eh_proj.disable_quantization)

    def test_router_weight_preserves_checkpoint_fp32_contract(self):
        weight_info = _weight_info()._get_weight_info()
        gate = next(
            component
            for module in weight_info.layer_weights[0]
            for component in module.get_components()
            if component.name == W.moe_gate
        )
        self.assertEqual(gate.data_type, torch.float32)

    def test_quantized_eh_projection_keeps_mxfp8_conversion_enabled(self):
        weight = _weight_info()
        names = {
            "language_model.model.mtp.layers.0.eh_proj.weight",
            "language_model.model.mtp.layers.0.eh_proj.weight_scale_inv",
        }
        weight._process_meta({}, names)

        eh_proj = next(
            item
            for item in weight._get_weight_info().weights
            if item.name == W.multi_tokens_predict_eh_proj
        )
        self.assertFalse(eh_proj.disable_quantization)

    def test_meta_accepts_bundled_and_standalone_layouts(self):
        bundled = _weight_info()
        bundled._process_meta(
            {},
            {
                "language_model.model.mtp.layers.0.enorm.weight",
                "language_model.model.mtp.layers.0.transformer_layer.block_sparse_moe.e_score_correction_bias",
            },
        )
        self.assertEqual(bundled.prefix, "language_model.")

        standalone = _weight_info()
        standalone._process_meta({}, {"model.mtp.layers.0.enorm.weight"})
        self.assertEqual(standalone.prefix, "")
        self.assertEqual(standalone._mtp_root, "model.mtp.layers.0.")

    def test_meta_rejects_missing_or_multiple_modules(self):
        with self.assertRaisesRegex(ValueError, "exactly module 0"):
            _weight_info()._process_meta({}, set())
        with self.assertRaisesRegex(ValueError, r"found modules \[0, 1\]"):
            _weight_info()._process_meta(
                {},
                {
                    "model.mtp.layers.0.enorm.weight",
                    "model.mtp.layers.1.enorm.weight",
                },
            )


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_inputs = None

    def forward(self, hidden_states, residual, fmha_impl, attn_inputs=None, **kwargs):
        self.attention_inputs = attn_inputs
        return SimpleNamespace(
            hidden_states=hidden_states + 1,
            residual=residual + 2,
            topk_indices=None,
        )


class _FakeFinalNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual + 10, None


class _CaptureGate(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8, 2, dtype=dtype))
        self.input_dtype = None

    def forward(self, hidden_states):
        self.input_dtype = hidden_states.dtype
        return hidden_states @ self.weight


class MiniMaxM3MTPForwardTest(unittest.TestCase):
    def test_fp32_router_weight_upcasts_bf16_activations(self):
        layer = object.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        layer.gate = _CaptureGate(torch.float32)

        hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)
        logits = layer._compute_router_logits(hidden_states)

        self.assertEqual(layer.gate.input_dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)

    def test_cp_prefill_masks_only_global_position_zero(self):
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        embeds = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 1
        fmha_impl = SimpleNamespace(
            fmha_params=SimpleNamespace(
                # The generic CP FMHA metadata is rank-local and therefore
                # incorrectly starts every rank at zero.
                positions_d=torch.arange(4, dtype=torch.int32)
            )
        )

        rank_zero_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=torch.tensor([0, 1, 14, 15], dtype=torch.int32),
                prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
            ),
        )
        masked = model._mask_position_zero_embeddings(
            embeds, fmha_impl, rank_zero_inputs
        )
        torch.testing.assert_close(masked[0], torch.zeros_like(masked[0]))
        torch.testing.assert_close(masked[1:], embeds[1:])

        nonzero_rank_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=torch.tensor([4, 5, 10, 11], dtype=torch.int32),
                prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
            ),
        )
        unmasked = model._mask_position_zero_embeddings(
            embeds, fmha_impl, nonzero_rank_inputs
        )
        torch.testing.assert_close(unmasked, embeds)

    def test_cp_prefill_prefix_does_not_mask_padding_or_position_zero(self):
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        embeds = torch.arange(32, dtype=torch.float32).reshape(4, 8) + 1
        fmha_impl = SimpleNamespace(
            fmha_params=SimpleNamespace(
                positions_d=torch.tensor([5, 6, 7, 8], dtype=torch.int32)
            )
        )
        attention_inputs = SimpleNamespace(
            prefix_lengths=torch.tensor([5], dtype=torch.int32),
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
                prefill_cp_chunk_lengths=torch.tensor([4], dtype=torch.int32),
            ),
        )

        actual = model._mask_position_zero_embeddings(
            embeds, fmha_impl, attention_inputs
        )
        torch.testing.assert_close(actual, embeds)

    def test_cp_mixed_batch_preserves_decode_mask_and_remaps_prefill_mask(self):
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        embeds = torch.arange(56, dtype=torch.float32).reshape(7, 8) + 1
        fmha_impl = SimpleNamespace(
            fmha_params=SimpleNamespace(
                # Decode rows are first.  The prefill positions below are
                # rank-local and deliberately contain spurious zeros.
                positions_d=torch.tensor([0, 9, 0, 1, 0, 1, 2], dtype=torch.int32)
            )
        )
        attention_inputs = SimpleNamespace(
            # Two decode streams followed by two prefill streams.  Only the
            # first prefill has no prefix and owns global position zero.
            prefix_lengths=torch.tensor([8, 10, 0, 5], dtype=torch.int32),
            context_parallel_info=SimpleNamespace(
                prefill_shuffle_indices=torch.tensor(
                    [0, 7, 0, 1, -1], dtype=torch.int32
                ),
                prefill_cp_chunk_lengths=torch.tensor([2, 3], dtype=torch.int32),
            ),
        )

        actual = model._mask_position_zero_embeddings(
            embeds, fmha_impl, attention_inputs
        )

        # Decode position zero remains masked, and only the true global
        # position zero of the prefix-free prefill is additionally masked.
        expected = embeds.clone()
        expected[0].zero_()
        expected[2].zero_()
        torch.testing.assert_close(actual, expected)

    def test_draft_refresh_skips_position_and_cp_mask_metadata(self):
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        embeds = torch.arange(24, dtype=torch.float32).reshape(3, 8) + 1

        class FailIfPositionMetadataIsRead:
            @property
            def fmha_params(self):
                raise AssertionError("draft refresh must not read position metadata")

        attention_inputs = SimpleNamespace(is_mtp_draft_prefill=True)
        actual = model._mask_position_zero_embeddings(
            embeds, FailIfPositionMetadataIsRead(), attention_inputs
        )

        self.assertIs(actual, embeds)

    def test_random_fusion_and_msa_metadata_contract(self):
        torch.manual_seed(20260809)
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Embedding(16, 8)
        model.pre_fc_norm_embedding = nn.Identity()
        model.pre_fc_norm_hidden = nn.Identity()
        model.fc = nn.Linear(16, 8, bias=False)
        layer = _FakeDecoderLayer()
        model.layers = nn.ModuleList([layer])
        model.norm = _FakeFinalNorm()
        model.layer_num = 1
        model.kv_cache = None
        model._share_mtp_topk_indices = False
        model._mtp_iteration_topk_buffers = [None]
        model._mtp_iteration_topk_valid_tokens = [0]
        model._mtp_iteration_topk_indices = [None]
        model._mtp_recurrent_hidden_states = None

        input_ids = torch.tensor([3, 5])
        previous_hidden = torch.randn(2, 8)
        positions = torch.tensor([1, 7])
        attention_inputs = SimpleNamespace(mtp_iteration_step=0, is_prefill=False)
        inputs = SimpleNamespace(
            input_ids=input_ids,
            input_hiddens=previous_hidden,
            attention_inputs=attention_inputs,
        )
        fmha_impl = SimpleNamespace(fmha_params=SimpleNamespace(positions_d=positions))

        embedded = model.embed_tokens(input_ids)
        expected_fused = model.fc(torch.cat([embedded, previous_hidden], dim=-1))
        with patch(
            "rtp_llm.models_py.model_desc.generic_moe_mtp.select_block_map_for_layer"
        ):
            output = model.forward(inputs, fmha_impl)

        expected_recurrent = expected_fused + 1 + torch.full_like(expected_fused, 2)
        torch.testing.assert_close(output.hidden_states, expected_recurrent + 10)
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(-1), expected_recurrent
        )
        torch.testing.assert_close(
            model.get_mtp_target_hidden_states(1), expected_recurrent[:1]
        )
        self.assertIs(layer.attention_inputs, attention_inputs)
        self.assertIs(MiniMaxM3MTPModel.decoder_layer_cls, MiniMaxM3MTPDecoderLayer)

    def test_recurrent_hidden_accessor_rejects_out_of_range_rows(self):
        model = object.__new__(MiniMaxM3MTPModel)
        nn.Module.__init__(model)
        model._mtp_recurrent_hidden_states = torch.zeros(2, 8)

        with self.assertRaisesRegex(RuntimeError, "more MTP recurrent hidden rows"):
            model.get_mtp_target_hidden_states(3)

    def test_speculative_contract_accepts_three_and_rejects_eight(self):
        target = SimpleNamespace(hidden_size=8, vocab_size=16)
        draft = SimpleNamespace(
            hidden_size=8,
            vocab_size=16,
            attn_config=SimpleNamespace(kv_cache_dtype=KvCacheDataType.FP8),
        )
        MiniMaxM3MTP.configure_speculative_model(
            SimpleNamespace(type=SpeculativeType.MTP, gen_num_per_cycle=3),
            target,
            draft,
        )
        with self.assertRaisesRegex(ValueError, "1..7"):
            MiniMaxM3MTP.configure_speculative_model(
                SimpleNamespace(type=SpeculativeType.MTP, gen_num_per_cycle=8),
                target,
                draft,
            )


if __name__ == "__main__":
    unittest.main()
