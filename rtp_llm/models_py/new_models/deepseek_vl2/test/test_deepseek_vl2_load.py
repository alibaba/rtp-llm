import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.deepseek_v3.attention import DeepSeekV32MlaAttention
from rtp_llm.models_py.new_models.deepseek_v3.moe import DeepSeekV32MoEBlock
from rtp_llm.models_py.new_models.deepseek_vl2.language import (
    DeepSeekVLV2Attention,
    DeepSeekVLV2ForCausalLM,
    _extract_config_values,
)
from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
    DeepSeekVLV2VisionModel,
    load_deepseek_vl2_vision,
    select_best_resolution,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import MlaOpsType

_RUN_LEGACY_TESTS = os.environ.get("RTP_LLM_RUN_DEEPSEEK_VL2_LEGACY_TESTS", "0") == "1"

if _RUN_LEGACY_TESTS:
    import math

    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models.deepseek_vl2.deepseek_vl2 import DeepSeekVLV2
    from rtp_llm.multimodal.multimodal_mixins.deepseek_vl2.deepseek_vl2_mixin import (
        DeepSeekVLV2ImageEmbedding,
        DeepSeekVLV2Mixin,
    )
    from rtp_llm.utils.base_model_datatypes import MMUrlType


def _parallelism(
    ep_size=1,
    tp_size=1,
    tp_rank=0,
    attn_tp_size=None,
    attn_tp_rank=None,
):
    resolved_attn_tp_size = tp_size if attn_tp_size is None else attn_tp_size
    resolved_attn_tp_rank = tp_rank if attn_tp_rank is None else attn_tp_rank
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
        get_attn_tp_size=lambda: resolved_attn_tp_size,
        get_attn_tp_rank=lambda: resolved_attn_tp_rank,
        get_ffn_tp_size=lambda: tp_size,
        get_ffn_tp_rank=lambda: tp_rank,
    )


def _load_config(
    ep_size=1,
    *,
    device="cpu",
    compute_dtype=torch.float32,
    tp_size=1,
    tp_rank=0,
    attn_tp_size=None,
    attn_tp_rank=None,
    lm_head_tp_size=None,
    lm_head_tp_rank=None,
):
    return NewLoaderConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        compute_dtype=compute_dtype,
        device=device,
        quant_config=QuantizationConfig("none"),
        parallelism_config=_parallelism(
            ep_size,
            tp_size,
            tp_rank,
            attn_tp_size,
            attn_tp_rank,
        ),
        moe_config=types.SimpleNamespace(fake_balance_expert=False),
        ep_size=ep_size,
        ep_rank=0,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        lm_head_tp_size=lm_head_tp_size,
        lm_head_tp_rank=lm_head_tp_rank,
    )


def _raw_language_config(*, use_mla=False, tie_word_embeddings=False):
    config = {
        "hidden_size": 4,
        "intermediate_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 8,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 2,
        "n_shared_experts": 1,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "scoring_func": "softmax",
        "routed_scaling_factor": 1.0,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": False,
        "topk_method": "greedy",
        "tie_word_embeddings": tie_word_embeddings,
        "use_mla": use_mla,
        "torch_dtype": "float32",
    }
    if use_mla:
        config.update(
            {
                "q_lora_rank": None,
                "kv_lora_rank": 2,
                "qk_nope_head_dim": 1,
                "qk_rope_head_dim": 2,
                "v_head_dim": 1,
            }
        )
    return config


def _model_config(
    model_path,
    *,
    use_mla=False,
    tie_word_embeddings=False,
    max_seq_len=32,
):
    return types.SimpleNamespace(
        model_type="deepseek_vl_v2",
        ckpt_path=model_path,
        hidden_size=4,
        num_layers=1,
        vocab_size=8,
        max_seq_len=max_seq_len,
        layernorm_eps=1e-6,
        expert_num=2,
        moe_k=1,
        moe_inter_size=2,
        inter_size=2,
        moe_style=2,
        moe_layer_index=[],
        scoring_func=0,
        activation_type="SiGLU",
        routed_scaling_factor=1.0,
        moe_n_group=1,
        moe_topk_group=1,
        has_moe_norm=False,
        enable_fp32_lm_head=False,
        tie_word_embeddings=tie_word_embeddings,
        eplb_config=types.SimpleNamespace(enable_eplb=False),
        mla_ops_type=MlaOpsType.AUTO,
        attn_config=types.SimpleNamespace(
            head_num=2,
            kv_head_num=2,
            size_per_head=2,
            use_mla=use_mla,
            q_lora_rank=0,
            kv_lora_rank=2 if use_mla else 0,
            nope_head_dim=1 if use_mla else 0,
            rope_head_dim=2 if use_mla else 0,
            v_head_dim=1 if use_mla else 0,
        ),
    )


def _model_config_for_raw(model_path, raw, *, use_mla):
    config = _model_config(model_path, use_mla=use_mla)
    config.hidden_size = raw.get("hidden_size", 4096)
    config.num_layers = raw.get("num_hidden_layers", 30)
    config.vocab_size = raw.get("vocab_size", 102400)
    config.attn_config.head_num = raw.get("num_attention_heads", 32)
    config.attn_config.kv_head_num = raw.get(
        "num_key_value_heads", config.attn_config.head_num
    )
    config.attn_config.size_per_head = config.hidden_size // config.attn_config.head_num
    q_lora_rank = raw.get("q_lora_rank", 1536)
    config.attn_config.q_lora_rank = 0 if q_lora_rank is None else q_lora_rank
    kv_lora_rank = raw.get("kv_lora_rank", 512)
    config.attn_config.kv_lora_rank = 0 if kv_lora_rank is None else kv_lora_rank
    config.attn_config.nope_head_dim = raw.get("qk_nope_head_dim", 128)
    config.attn_config.rope_head_dim = raw.get("qk_rope_head_dim", 64)
    config.attn_config.v_head_dim = raw.get("v_head_dim", 128)
    return config


def _write_config(model_path, raw_language):
    with open(f"{model_path}/config.json", "w", encoding="utf-8") as handle:
        json.dump({"language_config": raw_language}, handle)


def _dense_weights(*, use_mla=False, include_lm_head=True):
    prefix = "language.model."
    weights = {
        prefix
        + "embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(8, 4),
        prefix + "layers.0.input_layernorm.weight": torch.ones(4),
        prefix + "layers.0.post_attention_layernorm.weight": torch.ones(4),
        prefix
        + "layers.0.mlp.gate_proj.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(8, 4),
        prefix
        + "layers.0.mlp.up_proj.weight": torch.arange(
            32, 64, dtype=torch.float32
        ).reshape(8, 4),
        prefix
        + "layers.0.mlp.down_proj.weight": torch.arange(
            32, dtype=torch.float32
        ).reshape(4, 8),
        prefix + "norm.weight": torch.ones(4),
    }
    attention_prefix = prefix + "layers.0.self_attn."
    if use_mla:
        weights.update(
            {
                attention_prefix
                + "q_proj.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
                attention_prefix
                + "kv_a_proj_with_mqa.weight": torch.arange(
                    16, dtype=torch.float32
                ).reshape(4, 4),
                attention_prefix + "kv_a_layernorm.weight": torch.ones(2),
                attention_prefix
                + "kv_b_proj.weight": torch.arange(8, dtype=torch.float32).reshape(
                    4, 2
                ),
                attention_prefix
                + "o_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            }
        )
    else:
        weights.update(
            {
                attention_prefix
                + "q_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
                attention_prefix
                + "k_proj.weight": torch.arange(16, 32, dtype=torch.float32).reshape(
                    4, 4
                ),
                attention_prefix
                + "v_proj.weight": torch.arange(32, 48, dtype=torch.float32).reshape(
                    4, 4
                ),
                attention_prefix + "o_proj.weight": torch.eye(4),
            }
        )
    if include_lm_head:
        weights["language.lm_head.weight"] = torch.arange(
            32, 64, dtype=torch.float32
        ).reshape(8, 4)
    return weights


def _load_language(
    *,
    use_mla=False,
    tie_word_embeddings=False,
    mutate=None,
    max_seq_len=32,
    device="cpu",
    compute_dtype=torch.float32,
):
    model_path = tempfile.TemporaryDirectory()
    raw = _raw_language_config(use_mla=use_mla, tie_word_embeddings=tie_word_embeddings)
    _write_config(model_path.name, raw)
    weights = _dense_weights(use_mla=use_mla, include_lm_head=not tie_word_embeddings)
    if mutate is not None:
        mutate(weights)
    save_file(weights, f"{model_path.name}/model.safetensors")
    try:
        model = NewModelLoader(
            model_config=_model_config(
                model_path.name,
                use_mla=use_mla,
                tie_word_embeddings=tie_word_embeddings,
                max_seq_len=max_seq_len,
            ),
            load_config=_load_config(
                device=device,
                compute_dtype=compute_dtype,
            ),
            model_path=model_path.name,
        ).load()
    except Exception:
        model_path.cleanup()
        raise
    return model_path, model, weights


class _FakeVision(torch.nn.Module):
    def __init__(self, token_count=4):
        super().__init__()
        self.proj = torch.nn.Linear(3, 1152)
        self.token_count = token_count

    def forward_features(self, images):
        pooled = images.mean(dim=(-1, -2))
        features = self.proj(pooled)
        return features[:, None, :].repeat(1, self.token_count, 1)


def _vision_config():
    return {
        "vision_config": {
            "model_name": "siglip_so400m_patch14_384",
            "image_size": 384,
            "patch_size": 14,
            "width": 1152,
            "layers": 27,
            "heads": 16,
            "mlp_ratio": 3.7362,
        },
        "projector_config": {
            "projector_type": "downsample_mlp_gelu",
            "input_dim": 1152,
            "n_embed": 4,
            "depth": 2,
            "mlp_ratio": 1,
            "downsample_ratio": 2,
        },
        "candidate_resolutions": [[384, 384]],
        "tile_tag": "2D",
        "global_view_pos": "head",
    }


class DeepSeekVLV2NewloaderTest(unittest.TestCase):
    def test_registry_resolves_language_and_vision_models(self):
        from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
            DeepSeekVLV2ForVisionEmbedding,
        )

        self.assertIs(get_model_class("deepseek_vl_v2"), DeepSeekVLV2ForCausalLM)
        self.assertIs(
            get_model_class("deepseek_vl2_vision"),
            DeepSeekVLV2ForVisionEmbedding,
        )

    def test_mha_language_loader_filters_vision_and_maps_prefixes(self):
        def add_vision_weight(weights):
            weights["vision.not_a_language_tensor"] = torch.ones(1)

        model_path, model, weights = _load_language(mutate=add_vision_weight)
        self.addCleanup(model_path.cleanup)
        self.assertIsInstance(model.layers[0].self_attn, DeepSeekVLV2Attention)
        torch.testing.assert_close(
            model.embed_tokens.weight,
            weights["language.model.embed_tokens.weight"],
        )
        torch.testing.assert_close(
            model.lm_head.weight, weights["language.lm_head.weight"]
        )
        self.assertEqual(
            model.layers[0].self_attn.qkv_proj.prefix,
            "layers.0.self_attn.qkv_proj",
        )
        self.assertEqual(
            model.layers[0].self_attn.o_proj.prefix,
            "layers.0.self_attn.o_proj",
        )
        self.assertEqual(
            model.layers[0].mlp.gate_up_proj.prefix,
            "layers.0.mlp.gate_up_proj",
        )

    def test_mha_forward_matches_fused_qkv_reference_and_calls_fmha_once(self):
        model_path, model, weights = _load_language()
        self.addCleanup(model_path.cleanup)
        attention = model.layers[0].self_attn

        class _ReferenceFmha:
            def __init__(self):
                self.calls = []

            def forward(self, qkv, kv_cache, layer_idx):
                self.calls.append((qkv, kv_cache, layer_idx))
                return qkv[..., :4]

        hidden_states = torch.tensor(
            [[0.25, -0.5, 1.0, 2.0], [-1.0, 0.75, 0.5, -0.25]],
            dtype=torch.float32,
        )
        fmha = _ReferenceFmha()
        actual = attention(hidden_states, fmha)

        prefix = "language.model.layers.0.self_attn."
        expected_qkv = torch.nn.functional.linear(
            hidden_states,
            torch.cat(
                [
                    weights[prefix + "q_proj.weight"],
                    weights[prefix + "k_proj.weight"],
                    weights[prefix + "v_proj.weight"],
                ],
                dim=0,
            ),
        )
        expected = torch.nn.functional.linear(
            expected_qkv[..., :4],
            weights[prefix + "o_proj.weight"],
        )

        self.assertEqual(len(fmha.calls), 1)
        torch.testing.assert_close(fmha.calls[0][0], expected_qkv)
        self.assertEqual(fmha.calls[0][1:], (None, 0))
        torch.testing.assert_close(actual, expected)

    def test_mla_language_loader_uses_direct_q_projection(self):
        model_path, model, _ = _load_language(use_mla=True)
        self.addCleanup(model_path.cleanup)
        attention = model.layers[0].self_attn
        self.assertIsInstance(attention, DeepSeekV32MlaAttention)
        self.assertEqual(attention.q_lora_rank, 0)
        self.assertIsNone(attention.q_b_proj)
        self.assertEqual(
            attention.q_a_proj.prefix,
            "layers.0.self_attn.q_proj",
        )
        self.assertEqual(
            attention.kv_b_proj.prefix,
            "layers.0.self_attn.kv_b_proj",
        )
        model._ensure_mla_kernel_layout()
        self.assertIsNotNone(model._mla_kernel_layout)

    def test_mla_rope_cache_uses_runtime_max_seq_len(self):
        model_path, model, _ = _load_language(use_mla=True, max_seq_len=64)
        self.addCleanup(model_path.cleanup)
        self.assertEqual(model.cos_sin_cache.shape[0], 64)

    def test_mla_lifecycle_preserves_fp32_rope_and_releases_checkpoint_weights(self):
        model_path, model, _ = _load_language(use_mla=True)
        self.addCleanup(model_path.cleanup)
        source_cache = model.cos_sin_cache.detach().clone()
        model._ensure_mla_kernel_layout()

        model.to(dtype=torch.bfloat16)
        self.assertEqual(model.cos_sin_cache.dtype, torch.float32)
        torch.testing.assert_close(model.cos_sin_cache, source_cache)
        self.assertIs(
            model._mla_kernel_layout._cos_sin_cache,
            model.cos_sin_cache,
        )

        attention = model.layers[0].self_attn
        with mock.patch(
            "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
            return_value=True,
        ), mock.patch.object(attention, "release_checkpoint_only_weights") as release:
            self.assertTrue(model.initialize(object()))
        release.assert_called_once_with()

    def test_initialize_failure_does_not_build_or_release_mla_layout(self):
        model_path, model, _ = _load_language(use_mla=True)
        self.addCleanup(model_path.cleanup)
        attention = model.layers[0].self_attn
        with mock.patch(
            "rtp_llm.models_py.model_desc.module_base.GptModelBase.initialize",
            return_value=False,
        ), mock.patch.object(
            model, "_ensure_mla_kernel_layout"
        ) as ensure, mock.patch.object(
            attention, "release_checkpoint_only_weights"
        ) as release:
            self.assertFalse(model.initialize(object()))
        ensure.assert_not_called()
        release.assert_not_called()

    def test_mla_direct_q_forward_matches_reference_projections(self):
        model_path, model, weights = _load_language(use_mla=True)
        self.addCleanup(model_path.cleanup)
        attention = model.layers[0].self_attn

        class _ReferenceFmha:
            def forward(
                self,
                q,
                compressed_kv,
                k_pe,
                kv_cache,
                layer_idx,
                topk_indices,
            ):
                self.q = q
                self.compressed_kv = compressed_kv
                self.k_pe = k_pe
                self.call_metadata = (kv_cache, layer_idx, topk_indices)
                return q[..., :1]

        hidden_states = torch.tensor(
            [[0.25, -0.5, 1.0, 2.0], [-1.0, 0.75, 0.5, -0.25]],
            dtype=torch.float32,
        )
        fmha = _ReferenceFmha()
        actual = attention(hidden_states, fmha)

        prefix = "language.model.layers.0.self_attn."
        q = torch.nn.functional.linear(
            hidden_states,
            weights[prefix + "q_proj.weight"],
        ).view(2, 2, 3)
        kv_a = torch.nn.functional.linear(
            hidden_states,
            weights[prefix + "kv_a_proj_with_mqa.weight"],
        )
        compressed_kv, k_pe = torch.split(kv_a, [2, 2], dim=-1)
        compressed_kv = compressed_kv * torch.rsqrt(
            compressed_kv.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        expected = torch.nn.functional.linear(
            q[..., :1].reshape(2, 2),
            weights[prefix + "o_proj.weight"],
        )

        torch.testing.assert_close(fmha.q, q)
        torch.testing.assert_close(fmha.compressed_kv, compressed_kv)
        torch.testing.assert_close(fmha.k_pe, k_pe)
        self.assertEqual(fmha.call_metadata, (None, 0, None))
        torch.testing.assert_close(actual, expected)

    def test_unknown_and_missing_language_tensors_fail_fast(self):
        def add_unknown(weights):
            weights["language.model.layers.0.self_attn.typo.weight"] = torch.ones(1)

        with self.assertRaisesRegex(RuntimeError, "typo"):
            model_path, _, _ = _load_language(mutate=add_unknown)
            model_path.cleanup()

        def remove_q(weights):
            del weights["language.model.layers.0.self_attn.q_proj.weight"]

        with self.assertRaisesRegex(RuntimeError, "qkv_proj.*weight"):
            model_path, _, _ = _load_language(mutate=remove_q)
            model_path.cleanup()

        def add_extra_layer(weights):
            weights["language.model.layers.1.input_layernorm.weight"] = torch.ones(4)

        with self.assertRaisesRegex(RuntimeError, "layers.1"):
            model_path, _, _ = _load_language(mutate=add_extra_layer)
            model_path.cleanup()

        def add_unknown_root(weights):
            weights["language_typo.weight"] = torch.ones(1)

        with self.assertRaisesRegex(RuntimeError, "language_typo"):
            model_path, _, _ = _load_language(mutate=add_unknown_root)
            model_path.cleanup()

    def test_tied_lm_head_copies_embedding_when_checkpoint_omits_head(self):
        model_path, model, _ = _load_language(tie_word_embeddings=True)
        self.addCleanup(model_path.cleanup)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)

    def test_mla_noaux_moe_loads_all_routed_and_shared_weights(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=True)
            raw.update(
                {
                    "first_k_dense_replace": 0,
                    "scoring_func": "sigmoid",
                    "norm_topk_prob": True,
                    "topk_method": "noaux_tc",
                }
            )
            _write_config(model_path, raw)
            weights = _dense_weights(use_mla=True)
            for projection in ("gate_proj", "up_proj", "down_proj"):
                del weights[f"language.model.layers.0.mlp.{projection}.weight"]
            weights["language.model.layers.0.mlp.gate.weight"] = torch.arange(
                8, dtype=torch.float32
            ).reshape(2, 4)
            weights["language.model.layers.0.mlp.gate.e_score_correction_bias"] = (
                torch.tensor([0.25, -0.5], dtype=torch.float32)
            )
            for expert_id in range(2):
                offset = float(10 * expert_id)
                weights[
                    "language.model.layers.0.mlp.experts."
                    f"{expert_id}.gate_proj.weight"
                ] = torch.full((2, 4), 1.0 + offset)
                weights[
                    "language.model.layers.0.mlp.experts." f"{expert_id}.up_proj.weight"
                ] = torch.full((2, 4), 2.0 + offset)
                weights[
                    "language.model.layers.0.mlp.experts."
                    f"{expert_id}.down_proj.weight"
                ] = torch.full((4, 2), 3.0 + offset)
            weights["language.model.layers.0.mlp.shared_experts.gate_proj.weight"] = (
                torch.full((2, 4), 4.0)
            )
            weights["language.model.layers.0.mlp.shared_experts.up_proj.weight"] = (
                torch.full((2, 4), 5.0)
            )
            weights["language.model.layers.0.mlp.shared_experts.down_proj.weight"] = (
                torch.full((4, 2), 6.0)
            )

            model_config = _model_config(model_path, use_mla=True)
            model_config.moe_layer_index = [0]
            model_config.scoring_func = 1
            model_config.has_moe_norm = True
            with torch.device("cpu"):
                model = DeepSeekVLV2ForCausalLM(
                    model_config,
                    _load_config(),
                )
            model.load_weights(weights.items())
            NewModelLoader._validate_loaded_weights(model)

        moe = model.layers[0].mlp
        self.assertIsInstance(moe, DeepSeekV32MoEBlock)
        self.assertEqual(moe.experts.prefix, "layers.0.mlp.experts")
        self.assertEqual(
            moe.shared_experts.gate_up_proj.prefix,
            "layers.0.mlp.shared_experts.gate_up_proj",
        )
        self.assertEqual(moe.experts._loaded_count, 6)
        torch.testing.assert_close(
            moe.gate.e_score_correction_bias,
            torch.tensor([0.25, -0.5], dtype=torch.float32),
        )

    def test_multimodal_embedding_masks_placeholder_ids_before_injection(self):
        model_path, model, _ = _load_language()
        self.addCleanup(model_path.cleanup)
        feature = torch.tensor(
            [[101.0, 102.0, 103.0, 104.0], [201.0, 202.0, 203.0, 204.0]]
        )
        inputs = types.SimpleNamespace(
            input_ids=torch.tensor([1, 999, 999]),
            embedding_inputs=types.SimpleNamespace(
                text_tokens_mask=torch.tensor([True, False, False])
            ),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[feature],
                mm_features_locs=torch.tensor([1]),
            ),
        )
        embeddings = model._embed_inputs(inputs)
        torch.testing.assert_close(embeddings[0], model.embed_tokens.weight[1])
        torch.testing.assert_close(embeddings[1:], feature)

    def test_multimodal_features_require_locations(self):
        model_path, model, _ = _load_language()
        self.addCleanup(model_path.cleanup)
        inputs = types.SimpleNamespace(
            input_ids=torch.tensor([1]),
            embedding_inputs=types.SimpleNamespace(
                text_tokens_mask=torch.tensor([False])
            ),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[torch.ones(1, 4)],
                mm_features_locs=None,
            ),
        )
        with self.assertRaisesRegex(ValueError, "without mm_features_locs"):
            model._embed_inputs(inputs)

    def test_multimodal_features_require_text_token_mask(self):
        model_path, model, _ = _load_language()
        self.addCleanup(model_path.cleanup)
        inputs = types.SimpleNamespace(
            input_ids=torch.tensor([999]),
            embedding_inputs=types.SimpleNamespace(text_tokens_mask=None),
            multimodal_inputs=types.SimpleNamespace(
                multimodal_features=[torch.ones(1, 4)],
                mm_features_locs=torch.tensor([0]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "without text_tokens_mask"):
            model._embed_inputs(inputs)

    def test_text_only_embedding_does_not_require_multimodal_inputs(self):
        model_path, model, _ = _load_language()
        self.addCleanup(model_path.cleanup)
        input_ids = torch.tensor([1, 2])
        inputs = types.SimpleNamespace(
            input_ids=input_ids,
            embedding_inputs=None,
            multimodal_inputs=None,
        )
        torch.testing.assert_close(
            model._embed_inputs(inputs),
            model.embed_tokens(input_ids),
        )

    def test_config_distinguishes_tiny_mha_and_default_mla_variants(self):
        with tempfile.TemporaryDirectory() as model_path:
            tiny = _extract_config_values(
                _model_config(model_path, use_mla=False),
                _load_config(),
                _raw_language_config(use_mla=False),
            )
            self.assertFalse(tiny["use_mla"])
            self.assertEqual(tiny["head_dim"], 2)

            raw_mla = _raw_language_config(use_mla=True)
            del raw_mla["use_mla"]
            mla = _extract_config_values(
                _model_config(model_path, use_mla=True),
                _load_config(),
                raw_mla,
            )
            self.assertTrue(mla["use_mla"])
            self.assertEqual(mla["q_lora_rank"], 0)
            self.assertEqual(mla["moe_layer_index"], [])

    def test_model_config_layer_override_is_authoritative_and_topology_is_checked(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=False)
            raw["num_hidden_layers"] = 2
            config = _model_config(model_path, use_mla=False)
            actual = _extract_config_values(config, _load_config(), raw)
            self.assertEqual(actual["num_layers"], 1)

            config.hidden_size = 8
            with self.assertRaisesRegex(ValueError, "hidden_size mismatch"):
                _extract_config_values(config, _load_config(), raw)

    def test_layer_override_filters_only_declared_truncated_layers(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=False)
            raw["num_hidden_layers"] = 2
            _write_config(model_path, raw)
            weights = _dense_weights(use_mla=False)
            weights["language.model.layers.1.input_layernorm.weight"] = torch.ones(4)
            save_file(weights, f"{model_path}/model.safetensors")

            with torch.device("cpu"):
                model = NewModelLoader(
                    model_config=_model_config(model_path, use_mla=False),
                    load_config=_load_config(),
                    model_path=model_path,
                ).load()

            should_load = model.checkpoint_weight_name_filter()
            self.assertFalse(
                should_load("language.model.layers.1.input_layernorm.weight")
            )
            self.assertTrue(
                should_load("language.model.layers.2.input_layernorm.weight")
            )
            self.assertTrue(
                should_load("language.model.layers.invalid.input_layernorm.weight")
            )

    def test_official_sparse_configs_preserve_variant_defaults(self):
        variants = (
            (
                {
                    "hidden_size": 1280,
                    "intermediate_size": 6848,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 10,
                    "num_key_value_heads": 10,
                    "vocab_size": 129280,
                    "max_position_embeddings": 4096,
                    "n_routed_experts": 64,
                    "num_experts_per_tok": 6,
                    "moe_intermediate_size": 896,
                    "n_shared_experts": 2,
                    "first_k_dense_replace": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "topk_method": "greedy",
                    "q_lora_rank": None,
                    "kv_lora_rank": None,
                    "qk_nope_head_dim": 0,
                    "qk_rope_head_dim": 0,
                    "v_head_dim": 0,
                    "use_mla": False,
                },
                {
                    "use_mla": False,
                    "num_layers": 12,
                    "num_heads": 10,
                    "head_dim": 128,
                    "q_lora_rank": 0,
                    "kv_lora_rank": 0,
                    "correction_bias": False,
                },
            ),
            (
                {
                    "hidden_size": 2048,
                    "intermediate_size": 10944,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "vocab_size": 129280,
                    "max_position_embeddings": 4096,
                    "n_routed_experts": 64,
                    "num_experts_per_tok": 6,
                    "moe_intermediate_size": 1408,
                    "n_shared_experts": 2,
                    "first_k_dense_replace": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "topk_method": "greedy",
                    "q_lora_rank": None,
                },
                {
                    "use_mla": True,
                    "num_layers": 27,
                    "num_heads": 16,
                    "q_lora_rank": 0,
                    "kv_lora_rank": 512,
                    "correction_bias": False,
                },
            ),
            (
                {
                    "hidden_size": 2560,
                    "intermediate_size": 12288,
                    "vocab_size": 129280,
                    "max_position_embeddings": 4096,
                    "n_routed_experts": 72,
                    "num_experts_per_tok": 6,
                    "moe_intermediate_size": 1536,
                    "n_shared_experts": 2,
                    "first_k_dense_replace": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "topk_method": "noaux_tc",
                    "scoring_func": "sigmoid",
                    "norm_topk_prob": True,
                    "routed_scaling_factor": 2.0,
                    "q_lora_rank": None,
                },
                {
                    "use_mla": True,
                    "num_layers": 30,
                    "num_heads": 32,
                    "q_lora_rank": 0,
                    "kv_lora_rank": 512,
                    "correction_bias": True,
                    "scoring_func": 1,
                    "routed_scaling_factor": 2.0,
                },
            ),
        )
        with tempfile.TemporaryDirectory() as model_path:
            for raw, expected in variants:
                with self.subTest(hidden_size=raw["hidden_size"]):
                    actual = _extract_config_values(
                        _model_config_for_raw(
                            model_path,
                            raw,
                            use_mla=expected["use_mla"],
                        ),
                        _load_config(),
                        raw,
                    )
                    for name, value in expected.items():
                        self.assertEqual(actual[name], value)

    def test_sigmoid_without_topk_method_preserves_greedy_compatibility(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=True)
            raw["scoring_func"] = "sigmoid"
            raw["first_k_dense_replace"] = 0
            del raw["topk_method"]
            config = _extract_config_values(
                _model_config(model_path, use_mla=True),
                _load_config(),
                raw,
            )
            self.assertEqual(config["topk_method"], "greedy")
            self.assertFalse(config["correction_bias"])

    def test_eplb_and_mla_mha_fallback_fail_before_model_construction(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=True)
            config = _model_config(model_path, use_mla=True)
            config.eplb_config.enable_eplb = lambda: True
            with self.assertRaisesRegex(ValueError, "EPLB is not supported"):
                _extract_config_values(config, _load_config(), raw)

            config.eplb_config.enable_eplb = False
            config.mla_ops_type = MlaOpsType.MHA
            with self.assertRaisesRegex(ValueError, "expanded-MHA fallback"):
                _extract_config_values(config, _load_config(), raw)

            non_mla_raw = _raw_language_config(use_mla=False)
            non_mla_config = _model_config(model_path, use_mla=False)
            non_mla_config.mla_ops_type = MlaOpsType.MHA
            values = _extract_config_values(
                non_mla_config,
                _load_config(),
                non_mla_raw,
            )
            self.assertFalse(values["use_mla"])

    def test_mla_dimensions_must_match_model_config(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=True)
            fields = (
                ("q_lora_rank", 1),
                ("kv_lora_rank", 3),
                ("nope_head_dim", 2),
                ("rope_head_dim", 4),
                ("v_head_dim", 2),
            )
            for field, value in fields:
                config = _model_config(model_path, use_mla=True)
                setattr(config.attn_config, field, value)
                expected_name = {
                    "nope_head_dim": "qk_nope_head_dim",
                    "rope_head_dim": "qk_rope_head_dim",
                }.get(field, field)
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, f"{expected_name} mismatch"
                ):
                    _extract_config_values(config, _load_config(), raw)

    def test_tied_embeddings_merge_sources_and_require_matching_tp(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=False, tie_word_embeddings=False)
            config = _model_config(
                model_path,
                use_mla=False,
                tie_word_embeddings=True,
            )
            values = _extract_config_values(config, _load_config(), raw)
            self.assertTrue(values["tie_word_embeddings"])

            mismatch = _load_config(
                tp_size=2,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
                lm_head_tp_size=2,
                lm_head_tp_rank=0,
            )
            with self.assertRaisesRegex(ValueError, "matching attention and LM-head"):
                _extract_config_values(config, mismatch, raw)

    def test_group_limited_routing_rejects_invalid_partition_and_capacity(self):
        with tempfile.TemporaryDirectory() as model_path:
            raw = _raw_language_config(use_mla=True)
            raw.update(
                {
                    "first_k_dense_replace": 0,
                    "n_routed_experts": 6,
                    "n_group": 4,
                    "topk_group": 1,
                    "topk_method": "group_limited_greedy",
                }
            )
            with self.assertRaisesRegex(ValueError, "must be divisible"):
                _extract_config_values(
                    _model_config(model_path, use_mla=True),
                    _load_config(),
                    raw,
                )

            raw.update(
                {
                    "n_routed_experts": 8,
                    "n_group": 4,
                    "num_experts_per_tok": 3,
                }
            )
            with self.assertRaisesRegex(ValueError, "grouped capacity"):
                _extract_config_values(
                    _model_config(model_path, use_mla=True),
                    _load_config(),
                    raw,
                )

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_full_variant_uses_official_deepseek_v2_defaults(self):
        config = ModelConfig()
        config.max_seq_len = 8192
        DeepSeekVLV2._from_hf(
            config,
            {
                "language_config": {
                    "hidden_size": 2560,
                    "intermediate_size": 12288,
                    "moe_intermediate_size": 1536,
                    "n_routed_experts": 72,
                    "n_shared_experts": 2,
                    "num_experts_per_tok": 6,
                    "first_k_dense_replace": 1,
                    "q_lora_rank": None,
                    "topk_method": "noaux_tc",
                    "scoring_func": "sigmoid",
                    "norm_topk_prob": True,
                    "routed_scaling_factor": 2.0,
                    "vocab_size": 129280,
                    "max_position_embeddings": 4096,
                }
            },
        )
        self.assertEqual(config.num_layers, 30)
        self.assertEqual(config.model_type, "deepseek_vl_v2")
        self.assertEqual(config.attn_config.head_num, 32)
        self.assertEqual(config.attn_config.kv_head_num, 32)
        self.assertTrue(config.attn_config.use_mla)
        self.assertEqual(config.attn_config.q_lora_rank, 0)
        self.assertEqual(config.attn_config.kv_lora_rank, 512)
        self.assertEqual(config.attn_config.nope_head_dim, 128)
        self.assertEqual(config.attn_config.rope_head_dim, 64)
        self.assertEqual(config.attn_config.v_head_dim, 128)
        self.assertEqual(config.layernorm_eps, 1e-6)
        self.assertEqual(config.scoring_func, 1)
        self.assertTrue(config.has_moe_norm)
        self.assertEqual(config.routed_scaling_factor, 2.0)
        self.assertEqual(config.max_seq_len, 8192)

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_legacy_mla_config_preserves_rope_layout_and_yarn_scaling(self):
        config = ModelConfig()
        DeepSeekVLV2._from_hf(
            config,
            {
                "language_config": {
                    **_raw_language_config(use_mla=True),
                    "rope_interleave": False,
                    "indexer_rope_interleave": True,
                    "rope_scaling": {
                        "type": "yarn",
                        "factor": 4.0,
                        "original_max_position_embeddings": 32,
                        "beta_fast": 16,
                        "beta_slow": 2,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                    },
                }
            },
        )
        rope = config.attn_config.rope_config
        self.assertTrue(rope.is_neox_style)
        self.assertFalse(rope.indexer_is_neox_style)
        self.assertEqual(rope.scale, 4.0)
        self.assertEqual(rope.factor1, 2.0)
        self.assertEqual(rope.factor2, 16.0)
        self.assertEqual(rope.max_pos, 32)
        self.assertEqual(rope.mscale, 1.0)
        expected_softmax_mscale = 0.1 * math.log(4.0) + 1.0
        self.assertAlmostEqual(
            config.attn_config.softmax_extra_scale,
            expected_softmax_mscale * expected_softmax_mscale,
            places=6,
        )

    def test_checkpoint_filter_keeps_all_language_tensors_but_rejects_vision(self):
        model = object.__new__(DeepSeekVLV2ForCausalLM)
        torch.nn.Module.__init__(model)
        model.layers = torch.nn.ModuleList([torch.nn.Identity()])
        model._checkpoint_num_layers = 1
        should_load = model.checkpoint_weight_name_filter()
        self.assertTrue(should_load("language.model.layers.0.mlp.gate.weight"))
        self.assertTrue(should_load("language.model.unknown.weight"))
        self.assertTrue(should_load("language.lm_head.weight"))
        self.assertTrue(should_load("language.model.layers.1.mlp.gate.weight"))
        self.assertTrue(should_load("unknown_root.weight"))
        self.assertFalse(should_load("vision.blocks.0.attn.qkv.weight"))
        self.assertFalse(should_load("projector.layers.0.weight"))
        self.assertFalse(should_load("image_newline"))
        self.assertFalse(should_load("view_seperator"))

    def test_vision_loader_is_complete_and_filters_language_tensors(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            expected = DeepSeekVLV2VisionModel(config, torch.float32)
        weights = {
            name: tensor.detach().clone()
            for name, tensor in expected.state_dict().items()
        }
        weights["language.model.embed_tokens.weight"] = torch.ones(2, 2)
        with tempfile.TemporaryDirectory() as model_path:
            save_file(weights, f"{model_path}/model.safetensors")
            with mock.patch(
                "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
                side_effect=lambda *args, **kwargs: _FakeVision(),
            ):
                loaded = load_deepseek_vl2_vision(
                    vision_config=config,
                    model_path=model_path,
                    compute_dtype=torch.float32,
                    device="cpu",
                )
        for name, tensor in expected.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor)

    def test_vision_unknown_and_missing_tensors_fail_fast(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            expected = DeepSeekVLV2VisionModel(config, torch.float32)
        weights = {
            name: tensor.detach().clone()
            for name, tensor in expected.state_dict().items()
        }

        for mutate, error in (
            (
                lambda values: values.update({"vision.typo": torch.ones(1)}),
                "typo",
            ),
            (lambda values: values.pop("image_newline"), "image_newline"),
        ):
            current = dict(weights)
            mutate(current)
            with self.subTest(error=error), tempfile.TemporaryDirectory() as model_path:
                save_file(current, f"{model_path}/model.safetensors")
                with mock.patch(
                    "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
                    side_effect=lambda *args, **kwargs: _FakeVision(),
                ):
                    with self.assertRaisesRegex(RuntimeError, error):
                        load_deepseek_vl2_vision(
                            vision_config=config,
                            model_path=model_path,
                            compute_dtype=torch.float32,
                            device="cpu",
                        )

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_image_embedding_composes_global_and_local_tiles(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        mm_params = types.SimpleNamespace(config=config)
        embedding = DeepSeekVLV2ImageEmbedding(mm_params, vision_model=vision)
        result, extra = embedding.embedding(
            [torch.ones(2, 3, 14, 14), 1, 1],
            mm_type=MMUrlType.IMAGE,
        )
        self.assertEqual(tuple(result.shape), (5, 4))
        self.assertIsNone(extra)

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_batched_image_embedding_supports_heterogeneous_tile_grids(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        embedding = DeepSeekVLV2ImageEmbedding(
            types.SimpleNamespace(config=config),
            vision_model=vision,
        )

        results = embedding.batched_embedding(
            [
                [torch.ones(2, 3, 2, 2), 1, 1],
                [torch.ones(3, 3, 2, 2), 2, 1],
            ],
            [MMUrlType.IMAGE, MMUrlType.DEFAULT],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(tuple(results[0][0].shape), (5, 4))
        self.assertEqual(tuple(results[1][0].shape), (6, 4))
        self.assertIsNone(results[0][1])
        self.assertIsNone(results[1][1])

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_image_embedding_preserves_two_by_three_tile_geometry(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(token_count=16),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        embedding = DeepSeekVLV2ImageEmbedding(
            types.SimpleNamespace(config=config),
            vision_model=vision,
        )
        result, extra = embedding.embedding(
            [torch.ones(7, 3, 14, 14), 2, 3],
            mm_type=MMUrlType.IMAGE,
        )

        self.assertEqual(tuple(result.shape), (37, 4))
        self.assertIsNone(extra)
        torch.testing.assert_close(result[6], embedding.view_seperator)
        for newline_index in (2, 5, 11, 16, 21, 26, 31, 36):
            torch.testing.assert_close(result[newline_index], embedding.image_newline)

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_image_embedding_rejects_video_and_mismatched_tile_count(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        embedding = DeepSeekVLV2ImageEmbedding(
            types.SimpleNamespace(config=config),
            vision_model=vision,
        )

        with self.assertRaisesRegex(ValueError, "image inputs only"):
            embedding.embedding(
                [torch.ones(2, 3, 2, 2), 1, 1],
                MMUrlType.VIDEO,
            )
        with self.assertRaisesRegex(ValueError, "must have shape"):
            embedding.embedding(
                [torch.ones(2, 3, 2, 2), 2, 1],
                MMUrlType.IMAGE,
            )

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_newloader_vision_route_owns_all_weights(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)

        class _Mixin(DeepSeekVLV2Mixin):
            pass

        model = object.__new__(_Mixin)
        model.use_new_loader = True
        model.mm_related_params = types.SimpleNamespace(
            config=config,
            vit_weights=object(),
        )
        model.ckpt_path = "/local/deepseek-vl2"
        model.compute_dtype = torch.float32
        model.device = "cpu"
        with mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.deepseek_vl2."
            "deepseek_vl2_mixin.load_deepseek_vl2_vision",
            return_value=vision,
        ) as loader:
            model._init_multimodal()
        loader.assert_called_once_with(
            vision_config=config,
            model_path="/local/deepseek-vl2",
            compute_dtype=torch.float32,
            device="cpu",
        )
        self.assertIs(model.mm_part.vision_model, vision)
        self.assertIsNone(model.mm_related_params.vit_weights)

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_legacy_vision_route_keeps_separator_weights(self):
        config = _vision_config()
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):

            class _Mixin(DeepSeekVLV2Mixin):
                pass

            model = object.__new__(_Mixin)
            model.use_new_loader = False
            model.mm_related_params = types.SimpleNamespace(
                config=config,
                vit_weights=None,
            )
            model._init_multimodal()

        weight_names = set(model.mm_related_params.vit_weights.weight_names)
        self.assertIn("image_newline", weight_names)
        self.assertIn("view_seperator", weight_names)
        self.assertTrue(any(name.startswith("vision.") for name in weight_names))
        self.assertTrue(any(name.startswith("projector.") for name in weight_names))

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_legacy_language_route_remains_available(self):
        model = object.__new__(DeepSeekVLV2)
        model.model_config = types.SimpleNamespace(
            attn_config=types.SimpleNamespace(use_mla=False)
        )
        model.parallelism_config = object()
        model.weight = object()
        model.moe_config = object()
        model.max_generate_batch_size = 1
        model.fmha_config = object()
        model.hw_kernel_config = object()
        model.device_resource_config = object()
        constructed = object()
        with mock.patch(
            "rtp_llm.models_py.model_desc.generic_moe.GenericMoeModel",
            return_value=constructed,
        ) as generic_model:
            model._create_python_model()
        self.assertIs(model.py_model, constructed)
        generic_model.assert_called_once()

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_legacy_language_route_rejects_unsupported_mla_layout(self):
        model = object.__new__(DeepSeekVLV2)
        model.model_config = types.SimpleNamespace(
            attn_config=types.SimpleNamespace(use_mla=True)
        )
        with self.assertRaisesRegex(RuntimeError, "USE_NEW_LOADER=1"):
            model._create_python_model()

    @unittest.skipUnless(_RUN_LEGACY_TESTS, "legacy compatibility target only")
    def test_resolution_selection_is_deterministic_and_validated(self):
        self.assertEqual(
            select_best_resolution((100, 200), [(384, 384), (384, 768)]),
            (384, 384),
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            select_best_resolution((100, 200), [])

        config = _vision_config()
        config["candidate_resolutions"] = [[15, 14]]
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        with self.assertRaisesRegex(ValueError, "divisible by image_size"):
            DeepSeekVLV2ImageEmbedding(
                types.SimpleNamespace(config=config),
                vision_model=vision,
            )

    def test_vision_rejects_config_that_disagrees_with_fixed_timm_topology(self):
        config = _vision_config()
        config["vision_config"].update(
            {
                "image_size": 384,
                "width": 1152,
                "layers": 26,
                "heads": 16,
                "mlp_ratio": 3.7362,
            }
        )
        with self.assertRaisesRegex(ValueError, "layers=27"):
            DeepSeekVLV2VisionModel(config, torch.float32)

    def test_vision_defaults_match_the_legacy_fixed_siglip_topology(self):
        config = _vision_config()
        config["vision_config"] = {}
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model",
            side_effect=lambda *args, **kwargs: _FakeVision(),
        ):
            vision = DeepSeekVLV2VisionModel(config, torch.float32)
        self.assertEqual(vision.vision_config.model_name, "siglip_so400m_patch14_384")
        self.assertEqual(vision.vision_config.patch_size, 14)
        self.assertEqual(vision.vision_config.width, 1152)
        self.assertEqual(vision.vision_config.layers, 27)
        self.assertEqual(vision.vision_config.mlp_ratio, 3.7362)


if __name__ == "__main__":
    unittest.main()
