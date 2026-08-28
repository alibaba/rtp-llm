import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.model_loader.linear_attn_weight import (
    split_kda_dt_bias,
    split_kda_qkv,
    split_kda_tp_dim1,
)
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models.glm5_3_flash import (
    Glm53Flash,
    Glm53FlashWeight,
    parse_glm53_flash_config,
)
from rtp_llm.models_py.model_desc import generic_moe, kimi_linear
from rtp_llm.models_py.modules.base.cuda import indexer_op
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _requires_prefill_cp_support,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashinfer_mla,
    flashmla_sparse_cp_impl,
    flashmla_sparse_impl,
    rope_emb_new,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import _activation_clamp
from rtp_llm.models_py.modules.hybrid import indexer as indexer_module
from rtp_llm.models_py.modules.hybrid.dense_mlp import ClampedSiluAndMul
from rtp_llm.models_py.triton_kernels.common.strided_slice_copy import (
    strided_slice_copy_,
)
from rtp_llm.ops import DataType, HWKernelConfig, HybridAttentionType, ParallelismConfig
from rtp_llm.utils.model_weight import W


def _test_config():
    layer_types = [
        "deepseek_sparse_attention" if (i + 1) % 4 == 0 else "linear_attention"
        for i in range(45)
    ]
    return {
        "text_config": {
            "dtype": "bfloat16",
            "hidden_size": 4096,
            "vocab_size": 154880,
            "max_position_embeddings": 1048576,
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "index_kpool": 4,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "indexer_rope_interleave": True,
            "index_share_for_mtp_iteration": True,
            "indexer_types": ["full"] * 45,
            "layer_types": layer_types,
            "linear_attn_config": {
                "num_heads": 64,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "intermediate_size": 12288,
            "moe_intermediate_size": 2048,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "scoring_func": "sigmoid",
            "routed_scaling_factor": 2.5,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-5,
        }
    }


class Glm53FlashConfigTest(unittest.TestCase):
    def test_mega_moe_activation_clamp_is_opt_in(self):
        self.assertEqual(_activation_clamp(SimpleNamespace(swiglu_limit=10.0)), 10.0)
        self.assertIsNone(_activation_clamp(SimpleNamespace(swiglu_limit=0.0)))

    def test_clamped_swiglu_matches_checkpoint_semantics(self):
        gate_up = torch.tensor([[12.0, -12.0, 12.0, -12.0]])
        actual = ClampedSiluAndMul(10.0)(gate_up)
        gate, up = gate_up.chunk(2, dim=-1)
        expected = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
            -10.0, 10.0
        )
        torch.testing.assert_close(actual, expected)

    def test_shared_expert_uses_checkpoint_swiglu_limit(self):
        config = parse_glm53_flash_config(_test_config())
        self.assertEqual(generic_moe._shared_expert_swiglu_limit(config), 10.0)

    def test_prefill_uses_safe_gate_for_bounded_gate(self):
        prefill = kimi_linear.KimiLinearKDAPrefill.__new__(
            kimi_linear.KimiLinearKDAPrefill
        )
        nn.Module.__init__(prefill)
        prefill.local_num_k_heads = 1
        prefill.local_num_v_heads = 1
        prefill.head_k_dim = 2
        prefill.head_v_dim = 2
        prefill.gate_lower_bound = -5.0
        prefill.alog = torch.zeros(1)
        prefill.dt_bias = torch.zeros(2)
        prefill._get_ssm_states = mock.Mock(return_value=None)
        attention_inputs = SimpleNamespace(
            input_lengths=torch.tensor([1], dtype=torch.int32),
            cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
            prefix_lengths=torch.tensor([0], dtype=torch.int32),
        )

        with mock.patch.object(
            kimi_linear,
            "chunk_kda",
            return_value=(
                torch.zeros(1, 1, 1, 2),
                None,
                None,
            ),
        ) as chunk:
            prefill._fla(
                torch.zeros(1, 6),
                torch.zeros(1, 2),
                torch.zeros(1, 1),
                None,
                64,
                attention_inputs,
            )

        self.assertTrue(chunk.call_args.kwargs["safe_gate"])
        self.assertEqual(chunk.call_args.kwargs["lower_bound"], -5.0)

    def test_decode_threads_gate_lower_bound_to_recurrence(self):
        decode = kimi_linear.KimiLinearKDADecode.__new__(
            kimi_linear.KimiLinearKDADecode
        )
        nn.Module.__init__(decode)
        decode.local_num_k_heads = 1
        decode.local_num_v_heads = 1
        decode.head_k_dim = 2
        decode.head_v_dim = 2
        decode.gate_lower_bound = -5.0
        decode.alog = torch.zeros(1)
        decode.dt_bias = torch.zeros(2)
        decode._get_bs_from_attention_input = mock.Mock(return_value=(1, 1))
        decode._get_ssm_states = mock.Mock(return_value=torch.zeros(1, 1, 2, 2))
        attention_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.zeros((1, 1), dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.ones(1, dtype=torch.int32),
        )

        with mock.patch.object(
            kimi_linear,
            "fused_recurrent_kda",
            return_value=(torch.zeros(1, 1, 1, 2), None),
        ) as recurrent:
            decode._fla(
                torch.zeros(1, 6),
                torch.zeros(1, 2),
                torch.zeros(1, 1),
                torch.empty(0),
                64,
                attention_inputs,
                False,
            )

        self.assertEqual(recurrent.call_args.kwargs["lower_bound"], -5.0)

    def test_decode_conv_does_not_receive_gate_lower_bound(self):
        decode = kimi_linear.KimiLinearKDADecode.__new__(
            kimi_linear.KimiLinearKDADecode
        )
        nn.Module.__init__(decode)
        decode.conv_weights = torch.ones(6, 4)
        decode.gate_lower_bound = -5.0
        decode._get_bs_from_attention_input = mock.Mock(return_value=(1, 1))
        decode._get_conv_states = mock.Mock(return_value=torch.zeros(1, 3, 6))
        attention_inputs = SimpleNamespace(
            kv_cache_kernel_block_id_device=torch.zeros((1, 1), dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.ones(1, dtype=torch.int32),
        )
        mixed_qkv = torch.ones(1, 6)

        with mock.patch.object(
            kimi_linear,
            "causal_conv1d_update",
            side_effect=lambda x, *args, **kwargs: x,
        ) as conv1d:
            output = decode._conv1d(
                mixed_qkv,
                torch.empty(0),
                64,
                attention_inputs,
                False,
            )

        self.assertTrue(torch.equal(output, mixed_qkv))
        self.assertNotIn("lower_bound", conv1d.call_args.kwargs)

    def test_glm_mla_uses_indexer_specific_norm_epsilon(self):
        config = SimpleNamespace(
            hybrid_attention_config=SimpleNamespace(
                hybrid_attention_types=[HybridAttentionType.NONE]
            ),
            quant_config=None,
            attn_config=object(),
            layernorm_eps=1e-5,
            moe_layer_index=[],
            activation_type="SiLU",
            hc_mult=1,
        )
        parallelism = SimpleNamespace()
        weights = {
            W.pre_ln_gamma: torch.ones(1),
            W.post_ln_gamma: torch.ones(1),
        }

        with (
            mock.patch.object(
                kimi_linear, "MlaAttention", return_value=nn.Identity()
            ) as mla,
            mock.patch.object(kimi_linear, "DenseMLP", return_value=nn.Identity()),
            mock.patch.object(kimi_linear, "RMSResNorm", return_value=nn.Identity()),
        ):
            kimi_linear.KimiLinearDecoderLayer(
                config,
                parallelism,
                weights,
                {},
                0,
                None,
            )

        self.assertEqual(mla.call_args.kwargs["indexer_layernorm_eps"], 1e-6)
        self.assertEqual(mla.call_args.args[4], 1e-5)

    def test_decode_does_not_require_prefill_cp_support(self):
        self.assertFalse(
            _requires_prefill_cp_support(
                SimpleNamespace(is_prefill=False), use_decode_mla=False
            )
        )
        self.assertFalse(
            _requires_prefill_cp_support(
                SimpleNamespace(is_prefill=True), use_decode_mla=True
            )
        )
        self.assertTrue(
            _requires_prefill_cp_support(
                SimpleNamespace(is_prefill=True), use_decode_mla=False
            )
        )

    def test_parse_hybrid_config(self):
        config = parse_glm53_flash_config(_test_config(), "/model")
        self.assertEqual(config.ckpt_path, "/model")
        self.assertEqual(config.model_type, "glm5_3_flash")
        self.assertEqual(config.num_layers, 45)
        self.assertEqual(config.moe_layer_index, list(range(3, 45)))
        self.assertEqual(config.attn_config.rope_head_dim, 0)
        self.assertFalse(config.has_positional_encoding)
        self.assertEqual(config.attn_config.indexer_topk, 512)
        self.assertEqual(config.attn_config.indexer_compress_ratio, 4)
        self.assertEqual(config.attn_config.indexer_compressor_overlap, 0)
        self.assertEqual(config.attn_config.sparse_attention_topk, 2051)
        self.assertEqual(config.attn_config.indexer_layer_ids, list(range(3, 45, 4)))
        self.assertEqual(
            config.hybrid_attention_config.hybrid_attention_types[0],
            HybridAttentionType.LINEAR,
        )
        self.assertEqual(
            config.hybrid_attention_config.hybrid_attention_types[3],
            HybridAttentionType.NONE,
        )
        self.assertEqual(config.linear_attention_config.linear_num_key_heads, 64)
        self.assertEqual(config.linear_attention_config.linear_key_head_dim, 128)
        self.assertEqual(config.linear_attention_config.linear_conv_kernel_dim, 4)
        self.assertEqual(config.kda_gate_lower_bound, -5.0)
        self.assertEqual(config.hc_mult, 4)
        self.assertTrue(
            config.hybrid_attention_config.enable_independent_kv_cache_pools
        )

    def test_rejects_disabled_kpool_compression(self):
        config_json = _test_config()
        config_json["text_config"]["index_kpool_compress"] = False
        with self.assertRaisesRegex(ValueError, "index_kpool_compress=true"):
            parse_glm53_flash_config(config_json)

    def test_rejects_wrong_kpool_ratio(self):
        config_json = _test_config()
        config_json["text_config"]["index_kpool"] = 2
        with self.assertRaisesRegex(ValueError, "index_kpool=4"):
            parse_glm53_flash_config(config_json)

    def test_rejects_raw_topk_not_divisible_by_kpool(self):
        config_json = _test_config()
        config_json["text_config"]["index_topk"] = 2047
        with self.assertRaisesRegex(ValueError, "positive and divisible"):
            parse_glm53_flash_config(config_json)

    def test_rejects_disabled_tail_selection(self):
        config_json = _test_config()
        config_json["text_config"]["index_kpool_always_select_tail"] = False
        with self.assertRaisesRegex(ValueError, "always_select_tail=true"):
            parse_glm53_flash_config(config_json)

    def test_kda_recurrent_state_is_always_fp32(self):
        config = parse_glm53_flash_config(_test_config())
        kv_cache_config = KVCacheConfig()
        kv_cache_config.ssm_state_dtype = "bf16"

        config.init_linear_attention_cache_precision(kv_cache_config)

        self.assertEqual(
            config.linear_attention_config.ssm_state_dtype,
            DataType.TYPE_FP32,
        )

    def test_parse_multimodal_config(self):
        config_json = _test_config()
        config_json.update(
            {
                "vision_config": {"hidden_size": 16},
                "image_start_token_id": 11,
                "image_end_token_id": 12,
                "video_start_token_id": 13,
                "video_end_token_id": 14,
            }
        )
        with (
            mock.patch("builtins.open", mock.mock_open(read_data="{}")),
            mock.patch("os.path.exists", return_value=True),
        ):
            config = parse_glm53_flash_config(config_json, "/model")
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[11, 12]])
        self.assertEqual(
            config.mm_related_params.special_tokens["default_mm_token"],
            "<|begin_of_image|><|image|><|end_of_image|>",
        )
        self.assertEqual(
            config.mm_related_params.config["vision_config"],
            {"hidden_size": 16, "rms_norm_eps": 1e-6},
        )

    def test_rejects_mismatched_layer_schedule(self):
        config_json = _test_config()
        config_json["text_config"]["layer_types"].pop()
        with self.assertRaisesRegex(ValueError, "must match num_hidden_layers"):
            parse_glm53_flash_config(config_json)

    def test_list_eos_tokens(self):
        config_json = _test_config()
        config_json["text_config"]["eos_token_id"] = [154820, 154827, 154829]
        config_json["text_config"]["pad_token_id"] = 154820
        config = parse_glm53_flash_config(config_json)
        self.assertEqual(config.special_tokens.eos_token_id, 154820)
        self.assertEqual(
            config.special_tokens.stop_words_id_list,
            [[154820], [154827], [154829]],
        )

    def test_sparse_mla_receives_global_weights(self):
        config = SimpleNamespace(
            hybrid_attention_config=SimpleNamespace(
                hybrid_attention_types=[HybridAttentionType.NONE]
            ),
            quant_config=None,
            attn_config=object(),
            layernorm_eps=1e-5,
            moe_layer_index=[],
            activation_type="SiGLU",
            hc_mult=1,
        )
        layer_weights = {W.pre_ln_gamma: object(), W.post_ln_gamma: object()}
        global_weights = {W.rope_cos_sin_cache: object()}
        kernel_config = object()

        with (
            mock.patch.object(kimi_linear, "MlaAttention") as mla_attention,
            mock.patch.object(kimi_linear, "DenseMLP"),
            mock.patch.object(kimi_linear, "RMSResNorm"),
        ):
            kimi_linear.KimiLinearDecoderLayer(
                config,
                object(),
                layer_weights,
                global_weights,
                0,
                object(),
                hw_kernel_config=kernel_config,
            )

        self.assertIs(mla_attention.call_args.kwargs["global_weights"], global_weights)
        self.assertIs(mla_attention.call_args.kwargs["hw_kernel_config"], kernel_config)

    def test_mla_workspace_does_not_require_layer_zero_mla_weights(self):
        old_workspace = flashinfer_mla.g_workspace_buffer
        flashinfer_mla.g_workspace_buffer = None
        try:
            with (
                mock.patch.object(torch.cuda, "current_device", return_value=0),
                mock.patch.object(torch, "empty", return_value=object()) as empty,
                mock.patch.object(
                    flashinfer_mla, "BatchPrefillWithRaggedKVCacheWrapper"
                ),
            ):
                flashinfer_mla.MlaFlashInferPrefillOp(
                    num_heads=1,
                    kv_lora_rank=1,
                    qk_rope_head_dim=0,
                    qk_nope_head_dim=1,
                    v_head_dim=1,
                    page_size=1,
                    softmax_extra_scale=1.0,
                    use_mla=True,
                    weights=[{}],
                )

            self.assertEqual(empty.call_args.kwargs["device"], torch.device("cuda:0"))
        finally:
            flashinfer_mla.g_workspace_buffer = old_workspace

    def test_nope_mla_and_indexer_skip_rotary_kernel(self):
        with mock.patch.object(
            flashmla_sparse_impl, "fuse_kernels_enabled", return_value=True
        ):
            self.assertFalse(
                flashmla_sparse_impl._fused_qk_rope_cat_cache_mla_enabled(0)
            )
            self.assertTrue(
                flashmla_sparse_impl._fused_qk_rope_cat_cache_mla_enabled(64)
            )

        dst = torch.ones(2, 4, 16)
        strided_slice_copy_(dst, torch.empty(2, 4, 0), 16)
        self.assertTrue(torch.equal(dst, torch.ones_like(dst)))

        gathered_ckv = torch.empty(8, 512)
        gathered_k_pe = torch.empty(4, 2, 0)
        self.assertEqual(
            flashmla_sparse_cp_impl._reshape_gathered_k_pe(
                gathered_k_pe, gathered_ckv
            ).shape,
            (8, 0),
        )

        self.assertEqual(
            flashinfer_mla._reshape_rope_tensor(torch.empty(2, 0), 1, 0).shape,
            (2, 1, 0),
        )
        self.assertEqual(
            flashinfer_mla._reshape_rope_tensor(torch.empty(2, 0), 4, 0).shape,
            (2, 4, 0),
        )

        rotary = rope_emb_new.NewMlaRotaryEmbeddingOp(
            torch.empty(1, 0), is_neox_style=False
        )
        positions = torch.tensor([0, 1], dtype=torch.int32)
        with mock.patch.object(
            rope_emb_new.rope, "_apply_rope_pos_ids_cos_sin_cache"
        ) as apply_rope:
            rotary.forward(
                torch.empty(2, 4, 0),
                torch.empty(2, 0),
                SimpleNamespace(positions_d=positions),
            )
        apply_rope.assert_not_called()

        op = object.__new__(indexer_op.IndexerOp)
        nn.Module.__init__(op)
        op.index_head_dim = 128
        op.index_n_heads = 2
        op.rope_head_dim = 0
        op.cos_sin_cache = torch.empty(1, 0)
        op.is_neox_style = False
        q = torch.ones(2, 2, 128, dtype=torch.bfloat16)
        k = torch.ones(2, 128, dtype=torch.bfloat16)
        with (
            mock.patch.object(
                indexer_op, "_rotate_activation", side_effect=lambda x: x
            ),
            mock.patch.object(
                indexer_op.rope, "_apply_rope_pos_ids_cos_sin_cache"
            ) as apply_rope,
        ):
            query, key = op.apply_rope_and_rotate_q_k(q, k, positions)
        apply_rope.assert_not_called()
        self.assertIs(query, q)
        self.assertIs(key, k)

    def test_indexer_only_passes_scales_to_fp8_linear(self):
        class FakeLinear:
            def __init__(self):
                self.calls = []

            def __call__(self, value, **kwargs):
                self.calls.append((value, kwargs))
                return value

        class FakeFp8Linear(FakeLinear):
            pass

        bf16_input = object()
        fp8_input = object()
        input_scale = object()
        with mock.patch.object(indexer_module, "CudaFp8GEMMLinear", FakeFp8Linear):
            fp16_projection = FakeLinear()
            self.assertIs(
                indexer_module._project_with_optional_fp8(
                    fp16_projection, bf16_input, fp8_input, input_scale
                ),
                bf16_input,
            )
            self.assertEqual(fp16_projection.calls, [(bf16_input, {})])

            fp8_projection = FakeFp8Linear()
            self.assertIs(
                indexer_module._project_with_optional_fp8(
                    fp8_projection, bf16_input, fp8_input, input_scale
                ),
                fp8_input,
            )
            self.assertEqual(
                fp8_projection.calls,
                [(fp8_input, {"input_scales": input_scale})],
            )

    def test_kimi_mhc_forward_uses_base_model_config(self):
        model = object.__new__(kimi_linear.KimiLinearModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hc_mult=4)
        model.embed_tokens = mock.Mock(return_value=torch.ones(2, 3))
        model.layers = nn.ModuleList()
        model.hc_enabled = True
        model.norm = mock.Mock(side_effect=lambda hidden: hidden)
        model.kv_cache = None
        inputs = SimpleNamespace(
            input_ids=torch.tensor([1, 2]),
            attention_inputs=SimpleNamespace(
                is_prefill=False,
                is_target_verify=False,
            ),
        )
        fmha_impl = SimpleNamespace(fmha_params=None)

        output = model.forward(inputs, fmha_impl)

        self.assertEqual(tuple(output.hidden_states.shape), (2, 3))
        self.assertEqual(tuple(model.norm.call_args.args[0].shape), (2, 3))

    def test_kda_prefill_uses_current_prefix_lengths_field(self):
        prefill = object.__new__(kimi_linear.KimiLinearKDAPrefill)
        nn.Module.__init__(prefill)
        prefill.conv_weights = torch.ones(3, 1)
        mixed_qkv = torch.ones(2, 3)
        prefix_lengths = torch.tensor([0], dtype=torch.int32)
        attention_inputs = SimpleNamespace(
            cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
            kv_cache_kernel_block_id_device=torch.tensor([[1]], dtype=torch.int32),
            prefix_lengths=prefix_lengths,
        )

        with mock.patch.object(
            kimi_linear,
            "causal_conv1d_fn",
            return_value=mixed_qkv.transpose(0, 1),
        ) as conv1d:
            output = prefill._conv1d(
                mixed_qkv,
                None,
                64,
                attention_inputs,
            )

        self.assertTrue(torch.equal(output, mixed_qkv))
        self.assertIs(conv1d.call_args.kwargs["prefix_lengths"], prefix_lengths)

    def test_kda_context_parallel_keeps_full_attention_heads(self):
        linear_config = parse_glm53_flash_config(_test_config()).linear_attention_config
        parallelism = SimpleNamespace(
            tp_size=4,
            get_attn_tp_size=lambda: 1,
        )
        weights = {
            W.linear_attn_conv1d_w: torch.empty(1, 1, 1),
            W.linear_attn_dt_b_kda: torch.empty(1),
            W.linear_attn_alog: torch.empty(1),
        }

        kda = kimi_linear.KimiLinearKDABase(
            linear_config,
            parallelism,
            weights,
        )

        self.assertEqual(kda.local_num_k_heads, 64)
        self.assertEqual(kda.local_num_v_heads, 64)
        self.assertEqual(kda.qkv_size, 64 * 128 * 3)

    def test_kda_and_hc_checkpoint_manifest(self):
        config = parse_glm53_flash_config(_test_config())
        manifest = object.__new__(Glm53FlashWeight)
        manifest.model_config = config
        kda_weights = manifest._get_kda_weight_info()
        weights = kda_weights + manifest._get_hc_weight_info()
        for weight in weights:
            manifest._prefix_checkpoint_names(weight)

        checkpoint_names = {
            checkpoint.name
            for weight in weights
            for checkpoint in getattr(weight, "weights", [])
        }
        prefix = "model.language_model.layers.{i}."
        self.assertIn(prefix + "self_attn.q_proj.weight", checkpoint_names)
        self.assertIn(prefix + "self_attn.g_b_proj.weight", checkpoint_names)
        self.assertIn(prefix + "self_attn.dt_bias", checkpoint_names)
        self.assertIn(prefix + "hc_attn_base", checkpoint_names)
        self.assertIn(prefix + "hc_ffn_scale", checkpoint_names)
        self.assertEqual(len(checkpoint_names), 21)
        self.assertTrue(all(weight.quantization_disabled for weight in kda_weights))

        manifest._prefix_checkpoint_names(weights[0])
        checkpoint_names = {
            checkpoint.name
            for weight in weights
            for checkpoint in getattr(weight, "weights", [])
        }
        self.assertTrue(
            all(
                "model.language_model.language_model." not in name
                for name in checkpoint_names
            )
        )

    @unittest.skipUnless(
        os.environ.get("GLM5_CKPT_PATH"),
        "GLM5_CKPT_PATH is required for the real-checkpoint contract test",
    )
    def test_real_checkpoint_manifest(self):
        ckpt_path = os.environ["GLM5_CKPT_PATH"]
        config = Glm53Flash._create_config(ckpt_path)
        with open(
            os.path.join(ckpt_path, "model.safetensors.index.json"),
            encoding="utf-8",
        ) as reader:
            checkpoint_keys = set(json.load(reader)["weight_map"])

        parallelism = ParallelismConfig()
        parallelism.world_size = 1
        parallelism.local_world_size = 1
        manifest = Glm53FlashWeight(
            config,
            parallelism,
            HWKernelConfig(),
            KVCacheConfig(),
        )
        manifest._process_meta({}, checkpoint_keys)
        weight_info = manifest._get_weight_info()

        missing = []

        def check_weight(weight, layer_id=None):
            for checkpoint in getattr(weight, "weights", []):
                name = checkpoint.name
                if layer_id is not None:
                    name = name.format(i=layer_id, expert_id=0)
                if name not in checkpoint_keys:
                    missing.append(name)
            if isinstance(weight, CompositeWeight):
                for sub_weight in weight.sub_weights.values():
                    check_weight(sub_weight, layer_id)

        for weight in weight_info.weights:
            check_weight(weight)
        for layer_id, layer_weights in enumerate(weight_info.layer_weights):
            for weight in layer_weights:
                check_weight(weight, layer_id)

        self.assertFalse(
            missing,
            "missing checkpoint tensors:\n" + "\n".join(missing),
        )

    @unittest.skipUnless(
        os.environ.get("GLM5_CKPT_PATH"),
        "GLM5_CKPT_PATH is required for the quantized-manifest contract test",
    )
    def test_real_quantized_checkpoint_manifest(self):
        ckpt_path = os.environ["GLM5_CKPT_PATH"]
        config = Glm53Flash._create_config(ckpt_path)
        config.init_precision_config(KVCacheConfig(), "BF16")
        with open(
            os.path.join(ckpt_path, "model.safetensors.index.json"),
            encoding="utf-8",
        ) as reader:
            checkpoint_keys = set(json.load(reader)["weight_map"])

        parallelism = ParallelismConfig()
        parallelism.world_size = 1
        parallelism.local_world_size = 1
        manifest = Glm53FlashWeight(
            config,
            parallelism,
            HWKernelConfig(),
            KVCacheConfig(),
        )
        manifest._process_meta({}, checkpoint_keys)
        manifest = manifest.get_weight_info()

        missing = []

        def check_weight(weight, layer_id=None):
            for checkpoint in getattr(weight, "weights", []):
                name = checkpoint.name
                if layer_id is not None:
                    name = name.format(i=layer_id, expert_id=0)
                if name not in checkpoint_keys:
                    missing.append(name)
            if isinstance(weight, CompositeWeight):
                for sub_weight in weight.sub_weights.values():
                    check_weight(sub_weight, layer_id)

        for weight in manifest.weights:
            check_weight(weight)
        for layer_id, layer_weights in enumerate(manifest.layer_weights):
            for weight in layer_weights:
                check_weight(weight, layer_id)

        self.assertFalse(
            missing,
            "missing quantized checkpoint tensors:\n" + "\n".join(missing),
        )

    @unittest.skipUnless(
        os.environ.get("GLM5_CKPT_PATH"),
        "GLM5_CKPT_PATH is required for the real-tokenizer contract test",
    )
    def test_real_tokenizer(self):
        ckpt_path = os.environ["GLM5_CKPT_PATH"]
        tokenizer = TokenizerFactory.create(ckpt_path, ckpt_path, "glm5_3_flash")
        self.assertTrue(tokenizer.encode("hello"))
        self.assertEqual(
            tokenizer.encode("<|begin_of_image|><|image|><|end_of_image|>"),
            [154830, 154854, 154831],
        )


class Glm53FlashKdaWeightTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            linear_num_key_heads=64,
            linear_num_value_heads=64,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        )

    @staticmethod
    def _load_config(tp_size, tp_rank):
        return SimpleNamespace(tp_size=tp_size, tp_rank=tp_rank)

    def test_qkv_tp_sections_do_not_cross(self):
        hidden_size = 2
        section_width = 64 * 128
        sections = tuple(
            torch.arange(hidden_size * section_width, dtype=torch.float32)
            .reshape(hidden_size, section_width)
            .add_(section_id * 1_000_000)
            for section_id in range(3)
        )
        fused = torch.cat(sections, dim=1)

        for tp_size in (1, 2, 4, 8):
            local_width = section_width // tp_size
            for tp_rank in range(tp_size):
                actual = split_kda_qkv(
                    fused, self._load_config(tp_size, tp_rank), self.config
                )
                begin = tp_rank * local_width
                expected = torch.cat(
                    [section[:, begin : begin + local_width] for section in sections],
                    dim=1,
                )
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.equal(actual, expected))

    def test_tp4_head_projections_and_dt_bias(self):
        dt_bias = torch.arange(64 * 128, dtype=torch.float32).reshape(64, 128)
        projection = torch.arange(2 * 64 * 128, dtype=torch.float32).reshape(
            2, 64 * 128
        )
        for tp_rank in range(4):
            load_config = self._load_config(4, tp_rank)
            actual_bias = split_kda_dt_bias(dt_bias.flatten(), load_config, self.config)
            actual_projection = split_kda_tp_dim1(projection, load_config, self.config)
            head_slice = slice(tp_rank * 16, (tp_rank + 1) * 16)
            self.assertTrue(torch.equal(actual_bias, dt_bias[head_slice].flatten()))
            self.assertTrue(
                torch.equal(
                    actual_projection,
                    projection[:, head_slice.start * 128 : head_slice.stop * 128],
                )
            )


if __name__ == "__main__":
    unittest.main()
