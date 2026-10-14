"""gpt-oss config lifecycle tests.

Verifies the real ModelFactory.create_model_config order for both size
variants (20b / 120b): _create_config -> build_model_config ->
_apply_kv_cache_config -> _post_build_model_config, plus lazy registration
resolution and architecture lookup. Uses synthetic config.json dirs (no
weights), per the config-lifecycle contract: only parsing behavior is
under test here; loader/forward coverage lives elsewhere.
"""

import json
import os
import tempfile
from unittest import TestCase, main

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.py_config_modules import LoraConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.gpt_oss.gpt_oss import SCORING_FUNC_TOPK_SOFTMAX
from rtp_llm.ops import HybridAttentionType, ProfilingDebugLoggingConfig

# Real config.json field values (openai/gpt-oss-20b @ 6cee5e81 and
# gpt-oss-120b @ b5c939de), SOURCE_FACT.
COMMON_FIELDS = {
    "architectures": ["GptOssForCausalLM"],
    "attention_bias": True,
    "eos_token_id": 200002,
    "experts_per_token": 4,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2880,
    "intermediate_size": 2880,
    "max_position_embeddings": 131072,
    "model_type": "gpt_oss",
    "num_attention_heads": 64,
    "num_experts_per_tok": 4,
    "num_key_value_heads": 8,
    "pad_token_id": 199999,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "rope_type": "yarn",
        "truncate": False,
    },
    "rope_theta": 150000,
    "sliding_window": 128,
    "swiglu_limit": 7.0,
    "tie_word_embeddings": False,
    "vocab_size": 201088,
}

SIZES = {
    "20b": {
        "num_hidden_layers": 24,
        "num_local_experts": 32,
    },
    "120b": {
        "num_hidden_layers": 36,
        "num_local_experts": 128,
    },
}


def _layer_types(num_layers):
    return [
        "sliding_attention" if bool((i + 1) % 2) else "full_attention"
        for i in range(num_layers)
    ]


def _write_config_dir(size):
    fields = dict(COMMON_FIELDS)
    fields.update(SIZES[size])
    fields["num_experts"] = fields["num_local_experts"]
    fields["layer_types"] = _layer_types(fields["num_hidden_layers"])
    tmp = tempfile.mkdtemp(prefix=f"gpt_oss_{size}_")
    with open(os.path.join(tmp, "config.json"), "w") as writer:
        json.dump(fields, writer)
    return tmp


def _model_args(ckpt_path):
    args = ModelArgs()
    args.ckpt_path = ckpt_path
    args.model_type = "gpt_oss"
    return args


class GptOssConfigTest(TestCase):
    def test_lazy_registration_resolves(self):
        model_cls = ModelFactory.get_model_cls("gpt_oss")
        self.assertEqual(model_cls.__name__, "GptOss")

    def test_architecture_lookup(self):
        from rtp_llm.model_factory_register import ModelDict

        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(
                {"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss"}
            ),
            "gpt_oss",
        )

    def test_create_config_20b_values(self):
        cls = ModelFactory.get_model_cls("gpt_oss")
        config = cls._create_config(_write_config_dir("20b"))
        attn = config.attn_config
        self.assertEqual(config.num_layers, 24)
        self.assertEqual(config.hidden_size, 2880)
        # BF16-native checkpoint: an unset act_type must default the
        # unquantized tensors to BF16, never the framework-wide FP16
        # default (120b carries fp16-subnormal-range router/bias values
        # that the FP16 cast mangles).
        self.assertEqual(config.config_dtype, "bf16")
        self.assertEqual(config.vocab_size, 201088)
        self.assertEqual(config.max_seq_len, 131072)
        self.assertEqual(attn.head_num, 64)
        self.assertEqual(attn.kv_head_num, 8)
        self.assertEqual(attn.size_per_head, 64)
        self.assertEqual(attn.rope_head_dim, 64)
        self.assertEqual(attn.sliding_window, 128)
        self.assertFalse(attn.use_mla)
        # non-default rope values survive the parse
        rope = attn.rope_config
        self.assertEqual(int(rope.style), 5)  # RopeStyle.Yarn
        self.assertEqual(rope.base, 150000)
        self.assertEqual(rope.dim, 64)
        self.assertEqual(rope.scale, 32.0)
        self.assertEqual(rope.max_pos, 4096)
        self.assertEqual(rope.factor1, 1.0)
        self.assertEqual(rope.factor2, 32.0)
        # HF yarn default attention factor: 0.1*ln(32)+1
        self.assertAlmostEqual(rope.mscale, 1.34657359028, places=8)
        self.assertEqual(config.layernorm_eps, 1e-5)
        # string assignment is coerced to the NormType enum
        self.assertIn("rmsnorm", str(config.norm_type))
        self.assertFalse(config.qk_norm)
        # MoE: every layer routed, top-4, custom router semantics
        self.assertEqual(config.expert_num, 32)
        self.assertEqual(config.moe_k, 4)
        self.assertEqual(config.moe_inter_size, 2880)
        self.assertEqual(config.n_shared_experts, 0)
        self.assertEqual(config.moe_layer_index, list(range(24)))
        self.assertEqual(config.scoring_func, SCORING_FUNC_TOPK_SOFTMAX)
        self.assertEqual(config.swiglu_limit, 7.0)
        self.assertEqual(config.routed_scaling_factor, 1.0)
        self.assertFalse(config.has_moe_norm)
        # layer schedule: alternating sliding/full starting sliding at L0
        types = config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(types[0], HybridAttentionType.SLIDING_WINDOW)
        self.assertEqual(types[1], HybridAttentionType.NONE)
        self.assertEqual(
            types, [HybridAttentionType.SLIDING_WINDOW if i % 2 == 0 else HybridAttentionType.NONE for i in range(24)]
        )
        # uniform MHA pool: hybrid grouping stays off
        self.assertFalse(config.hybrid_attention_config.enable_hybrid_attention)

    def test_create_config_120b_values(self):
        cls = ModelFactory.get_model_cls("gpt_oss")
        config = cls._create_config(_write_config_dir("120b"))
        self.assertEqual(config.num_layers, 36)
        self.assertEqual(config.expert_num, 128)
        types = config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(len(types), 36)
        self.assertEqual(types[35], HybridAttentionType.NONE)  # even count -> last full

    def test_full_lifecycle_builds_uniform_kv_specs(self):
        cls = ModelFactory.get_model_cls("gpt_oss")
        ckpt = _write_config_dir("20b")
        model_config = ModelFactory.create_model_config(
            _model_args(ckpt),
            lora_config=LoraConfig(),
            kv_cache_config=KVCacheConfig(),
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        )
        # BaseModel._post_build_model_config fills uniform MHA descs
        self.assertTrue(len(model_config.kv_cache_spec_descs) >= 24)
        # hybrid types survive build_model_config
        types = model_config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(len(types), 24)
        self.assertEqual(types[0], HybridAttentionType.SLIDING_WINDOW)
        # model_name set by the factory
        self.assertEqual(model_config.model_name, "GptOss")

    def test_default_layer_types_when_missing(self):
        # transformers fills layer_types with the same [S, F] pattern when the
        # field is absent; the parser must not depend on the field existing.
        ckpt = _write_config_dir("20b")
        with open(os.path.join(ckpt, "config.json")) as reader:
            fields = json.load(reader)
        del fields["layer_types"]
        with open(os.path.join(ckpt, "config.json"), "w") as writer:
            json.dump(fields, writer)
        cls = ModelFactory.get_model_cls("gpt_oss")
        config = cls._create_config(ckpt)
        types = config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(
            types, [HybridAttentionType.SLIDING_WINDOW if i % 2 == 0 else HybridAttentionType.NONE for i in range(24)]
        )


if __name__ == "__main__":
    main()
