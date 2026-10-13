"""openai/gpt-oss-20b & gpt-oss-120b model entry.

Config-level size variants of one architecture (verified from HF configs,
SOURCE_FACT): same hidden 2880 / 64 Q heads / 8 KV heads / head_dim 64 /
vocab 201088 / MXFP4 routed experts top-4; different layer count (24 vs 36)
and expert count (32 vs 128).

Math contract (transformers modeling_gpt_oss, cross-checked against
checkpoint shapes):
  - GQA attention, BF16 q/k/v/o all with bias; no QK-norm, no softcapping.
  - Full head_dim RoPE (half-rotate), theta 150000, YaRN factor 32
    (beta_fast 32, beta_slow 1, original max 4096; attention scaling
    0.1*ln(32)+1 applied to cos/sin).
  - Per-head attention sinks: an extra denominator-only softmax column
    (numerator dropped).
  - Layer types alternate [sliding_attention(window=128), full_attention].
  - MoE every layer (no shared experts). Router: top-4 on raw logits
    (bias included), softmax over the 4 selected logits only.
  - Expert activation: gate=min(gate,7) (no lower bound), up=clamp(up,-7,7),
    act = gate*sigmoid(1.702*gate)*(up+1) — NOT silu(gate)*up.
  - Routed experts MXFP4: E2M1 nibble-packed + UE8M0 group-32 scales,
    gate/up rows interleaved in the checkpoint; experts carry biases.
"""

import json
import logging
import math
import os
from typing import Any, Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import HybridAttentionType

from .gpt_oss_weight import GptOssWeightInfo

# gpt-oss router semantics: rank on raw biased logits, then softmax over the
# selected top-k values. Extends the DeepSeek scoring_func int convention.
SCORING_FUNC_TOPK_SOFTMAX = 3


class GptOss(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        cls._parse_basic_config(config_json, config)
        cls._parse_rope_config(config_json, config)
        cls._parse_normalization_config(config_json, config)
        cls._parse_moe_config(config_json, config)
        cls._parse_layer_types(config_json, config)
        return config

    @classmethod
    def _parse_basic_config(cls, config_json: dict, config: ModelConfig):
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json["num_key_value_heads"]
        config.attn_config.size_per_head = config_json["head_dim"]
        # Full-dim RoPE; the nope/rope split helpers expect rope==head_dim.
        config.attn_config.nope_head_dim = 0
        config.attn_config.rope_head_dim = config.attn_config.size_per_head
        config.attn_config.v_head_dim = config.attn_config.size_per_head
        config.attn_config.use_mla = False
        config.attn_config.is_sparse = False
        config.attn_config.sliding_window = int(
            config_json.get("sliding_window", 0) or 0
        )
        # q/k/v/o biases exist — declared via W.attn_qkv_b/W.attn_o_b in the
        # weight info; consumers key off the loaded weights.
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.max_seq_len = config_json["max_position_embeddings"]
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)

    @classmethod
    def _parse_rope_config(cls, config_json: dict, config: ModelConfig):
        rope = config.attn_config.rope_config
        rope.style = 5  # yarn (half-rotate)
        rope.base = int(config_json.get("rope_theta", 150000))
        rope.dim = config.attn_config.size_per_head
        rope.offset = 0
        rope_scaling = config_json.get("rope_scaling") or {}
        if rope_scaling.get("rope_type") == "yarn":
            factor = float(rope_scaling.get("factor", 32.0))
            rope.scale = factor
            rope.max_pos = int(
                rope_scaling.get("original_max_position_embeddings", 4096)
            )
            rope.factor1 = float(rope_scaling.get("beta_slow", 1.0))
            rope.factor2 = float(rope_scaling.get("beta_fast", 32.0))
            rope.mscale = (0.1 * math.log(factor) + 1.0) if factor > 1 else 1.0
            rope.extrapolation_factor = 1.0
        else:
            logging.warning(
                "gpt-oss config has no yarn rope_scaling; using defaults"
            )
            rope.scale = 32.0
            rope.max_pos = 4096
            rope.factor1 = 1.0
            rope.factor2 = 32.0
            rope.mscale = 0.1 * math.log(32.0) + 1.0

    @classmethod
    def _parse_normalization_config(cls, config_json: dict, config: ModelConfig):
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-5)
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.qk_norm = False
        config.activation_type = "SiGLU"

    @classmethod
    def _parse_moe_config(cls, config_json: Dict[str, Any], config: ModelConfig):
        config.moe_k = int(config_json["num_experts_per_tok"])
        config.expert_num = int(config_json["num_local_experts"])
        config.moe_inter_size = int(config_json["intermediate_size"])
        config.inter_size = 0
        config.n_shared_experts = 0
        config.moe_style = 1  # every layer is MoE
        config.moe_layer_index = list(range(config.num_layers))
        # softmax over the k selected raw logits already sums to 1
        config.has_moe_norm = False
        config.routed_scaling_factor = 1.0
        config.scoring_func = SCORING_FUNC_TOPK_SOFTMAX
        config.swiglu_limit = float(config_json.get("swiglu_limit", 7.0))
        config.moe_n_group = 0
        config.moe_topk_group = 0
        config.num_hash_layers = 0
        # BF16 attention/router/embed/lm_head per modules_to_not_convert.
        # The checkpoint's tensors are bf16 (the configs omit torch_dtype);
        # default config_dtype to bf16 so an unset act_type does not route
        # the unquantized tensors through an fp16 cast — 120b carries
        # router/bias values in fp16's subnormal range that the cast
        # mangles (observed: expert-selection flips).
        config.config_dtype = config_json.get("torch_dtype") or "bf16"
        # Keep lm_head in checkpoint dtype (bf16), like DeepSeek V4.
        config.enable_fp32_lm_head = False

    @classmethod
    def _parse_layer_types(cls, config_json: dict, config: ModelConfig):
        """Store the per-layer sliding/full schedule for the python model.

        The C++ cache layout stays uniform MHA (kv_cache_spec_descs left
        empty so BaseModel's default hook builds one shared pool); SWA is
        applied as a windowed attention read, not a physical ring cache.
        """
        layer_types: List[str] = config_json.get("layer_types") or [
            "sliding_attention" if bool((i + 1) % 2) else "full_attention"
            for i in range(config.num_layers)
        ]
        hybrid_types: List[HybridAttentionType] = []
        for layer_type in layer_types[: config.num_layers]:
            if layer_type == "sliding_attention":
                hybrid_types.append(HybridAttentionType.SLIDING_WINDOW)
            else:
                hybrid_types.append(HybridAttentionType.NONE)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_types
        # Deliberately NOT setting enable_hybrid_attention: all layers share
        # the same MHA KV pool; only the attention window differs per layer.

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.gpt_oss import GptOssModel

        self.py_model = GptOssModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model

    def support_cuda_graph(self) -> bool:
        return True

    @staticmethod
    def get_weight_cls():
        return GptOssWeightInfo


register_model("gpt_oss", GptOss, ["GptOssForCausalLM"], ["openai/gpt-oss-20b", "openai/gpt-oss-120b"])
