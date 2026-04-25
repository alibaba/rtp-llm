"""DeepSeek-V4 model entry point (skeleton).

This file establishes the registration and HF config parsing for DeepseekV4.
For M0 the forward path is intentionally not wired up — `_create_python_model`
raises so the engine fails fast if you try to actually run inference. Subsequent
milestones (M1 mHC, M2 HCA, M3 CSA, M4 heterogeneous KV cache) flesh out the
real model body in `rtp_llm/models_py/model_desc/deepseek_v4_model.py`.

Key V4 architecture differences vs V3 — see
`/home/wangyin.yx/workspace/work_memory/plans/develop_ds_v4.md` for the full
breakdown:

  - MQA single-KV-head with head_dim=512 (no kv_lora_rank, no v_head_dim split)
  - Per-layer attention type from `compress_ratios`: 0=SWA-only, 4=CSA, 128=HCA
  - mHC residual: residual stream lives in [n_hc=4, hidden_size]
  - Grouped output projection: n_h heads -> g groups -> o_lora_rank -> hidden
  - Lightning indexer with FP4 QK / BF16 score for CSA (top-k selection)
  - Sliding-window bypass + per-head learnable attention sink in softmax denom
  - sqrt(softplus) MoE scoring (scoring_func=2), no n_group/topk_group
  - First `num_hash_layers` MoE layers route via deterministic token-id hash
  - SwiGLU clamping (linear in [-swiglu_limit,limit], gate <= limit)
  - Dual RoPE bases: rope_theta=10000 main, compress_rope_theta=160000 for
    compressed K branch
"""

import functools
import json
import logging
import os
from typing import List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import (
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import (
    DeepSeekV2,
    DeepSeekV2Weight,
    DeepSeekV3Mtp,
    DeepSeekV3MtpWeight,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    yarn_get_mscale,
    zeros,
)


SCORING_FUNC_SOFTMAX = 0
SCORING_FUNC_SIGMOID = 1
SCORING_FUNC_SQRT_SOFTPLUS = 2  # DeepSeek-V4


class DeepSeekV4Weight(DeepSeekV2Weight):
    """DeepSeek-V4 weight info.

    Declares only the GLOBAL weights the engine needs for logit projection +
    norm + embedding lookup. All per-layer weights (attention, MoE, mHC,
    compressor, indexer) are loaded inside `DeepSeekV4Model.initialize()` via
    a direct safetensors loader, bypassing the framework's AtomicWeight flow
    (V4-Flash has 67k+ tensors spread across FP4/FP8 packed shards).

    Checkpoint key layout (V4 official):
      embed.weight   -> W.embedding
      norm.weight    -> W.final_ln_gamma
      head.weight    -> W.lm_head
    """

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = [[] for _ in range(self._num_layers)]
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("embed.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("head.weight", identity)],
                identity,
            ),
        ]
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekV4(DeepSeekV2):
    """DeepSeek-V4 entry point.

    M0: parse HF config and register. Engine instantiation will fail at
    `_create_python_model` until M2 lands the HCA-only forward path.
    """

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = ModelConfig()
        config.attn_config.head_num = 0
        config.attn_config.kv_head_num = 0
        config.attn_config.size_per_head = 0
        config.num_layers = 0
        config.inter_size = 0
        config.vocab_size = 129280
        config.max_seq_len = 8192
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path)
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model

        self.py_model = DeepSeekV4Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )

    @staticmethod
    def _from_hf(config: ModelConfig, ckpt_path: str):  # noqa: C901  (acceptably long)
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return

        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        # ---- basic geometry ----
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get("num_key_value_heads", 1)

        # V4 is MQA with monolithic head_dim — NOT split into nope+rope.
        # Only the LAST `qk_rope_head_dim` dims of each head get RoPE.
        head_dim = config_json["head_dim"]
        rope_dim = config_json["qk_rope_head_dim"]
        config.attn_config.size_per_head = head_dim
        config.attn_config.nope_head_dim = head_dim - rope_dim  # for code that splits
        config.attn_config.rope_head_dim = rope_dim
        config.attn_config.v_head_dim = head_dim
        config.attn_config.use_mla = False  # V4 uses MQA, not MLA

        q_lora_rank = config_json.get("q_lora_rank")
        config.attn_config.q_lora_rank = int(q_lora_rank) if q_lora_rank else 0
        config.attn_config.kv_lora_rank = 0  # V4 has no KV LoRA

        # ---- RoPE: main base + separate compressed base ----
        config.attn_config.rope_config.base = int(config_json.get("rope_theta", 10000))
        config.attn_config.rope_config.dim = rope_dim
        config.attn_config.rope_config.offset = head_dim - rope_dim  # partial RoPE on tail
        config.attn_config.rope_config.style = 0  # interleaved by default per V4
        config.attn_config.compress_rope_theta = float(
            config_json.get("compress_rope_theta", 160000)
        )

        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            scaling_factor = rope_scaling["factor"]
            config.attn_config.rope_config.scale = scaling_factor
            config.attn_config.rope_config.factor1 = float(rope_scaling.get("beta_slow", 1))
            config.attn_config.rope_config.factor2 = float(rope_scaling.get("beta_fast", 32))
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            # V4 config.json doesn't carry mscale fields like V3; fall back to factor-of-2
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale

        # ---- per-layer attention schedule ----
        compress_ratios = config_json["compress_ratios"]
        # Drop the trailing MTP entry (always 0) so the schedule matches main layers.
        # Length should be num_layers (MTP layers count separately).
        if len(compress_ratios) == config.num_layers + config_json.get(
            "num_nextn_predict_layers", 0
        ):
            main_ratios = compress_ratios[: config.num_layers]
        else:
            main_ratios = compress_ratios
        config.attn_config.layer_compress_ratios = list(main_ratios)

        # ---- output projection / SWA / sink ----
        config.attn_config.o_groups = int(config_json["o_groups"])
        config.attn_config.o_lora_rank = int(config_json["o_lora_rank"])
        config.attn_config.sliding_window = int(config_json.get("sliding_window", 0))

        # ---- sparse indexer (CSA) ----
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = int(config_json["index_head_dim"])
        config.attn_config.indexer_head_num = int(config_json["index_n_heads"])
        config.attn_config.indexer_topk = int(config_json["index_topk"])

        # ---- MoE ----
        scoring_func = config_json.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = SCORING_FUNC_SOFTMAX
        elif scoring_func == "sigmoid":
            config.scoring_func = SCORING_FUNC_SIGMOID
        elif scoring_func == "sqrtsoftplus":
            config.scoring_func = SCORING_FUNC_SQRT_SOFTPLUS
        else:
            raise ValueError(f"Unknown V4 scoring_func: {scoring_func}")

        config.routed_scaling_factor = config_json["routed_scaling_factor"]
        config.moe_k = config_json["num_experts_per_tok"]
        config.expert_num = config_json["n_routed_experts"]
        moe_intermediate_size = config_json["moe_intermediate_size"]
        # V4 drops n_group/topk_group constraints; explicitly zero them so any
        # downstream code falls back to plain top-k.
        config.moe_n_group = 0
        config.moe_topk_group = 0
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        config.moe_style = 2  # shared + expert
        n_shared_experts = config_json["n_shared_experts"]
        config.inter_size = n_shared_experts * moe_intermediate_size
        # Every layer is MoE in V4 (compress_ratios doesn't include dense replacement)
        config.moe_layer_index = list(range(config.num_layers))

        # ---- V4-only model-level fields ----
        config.hc_mult = int(config_json.get("hc_mult", 0))
        config.hc_sinkhorn_iters = int(config_json.get("hc_sinkhorn_iters", 0))
        config.hc_eps = float(config_json.get("hc_eps", 1e-6))
        config.swiglu_limit = float(config_json.get("swiglu_limit", 0.0))
        config.num_hash_layers = int(config_json.get("num_hash_layers", 0))

        config.config_dtype = config_json.get("torch_dtype", None)

        logging.info(
            "DeepSeek-V4 config loaded: layers=%d hidden=%d head_num=%d head_dim=%d "
            "rope_dim=%d q_lora=%d compress_ratios=%s o_groups=%d sliding=%d "
            "n_experts=%d num_hash_layers=%d hc_mult=%d swiglu_limit=%g "
            "scoring_func=%d compress_rope_theta=%g",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            config.attn_config.size_per_head,
            config.attn_config.rope_head_dim,
            config.attn_config.q_lora_rank,
            list(config.attn_config.layer_compress_ratios)[:8] + ["..."],
            config.attn_config.o_groups,
            config.attn_config.sliding_window,
            config.expert_num,
            config.num_hash_layers,
            config.hc_mult,
            config.swiglu_limit,
            config.scoring_func,
            config.attn_config.compress_rope_theta,
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4Weight


class DeepSeekV4MtpWeight(DeepSeekV4Weight, DeepSeekV3MtpWeight):
    """MTP weight loader — reuses V3 MTP layout (enorm/hnorm/eh_proj/shared_head)."""


class DeepSeekV4Mtp(DeepSeekV4, DeepSeekV3Mtp):

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.moe_layer_index = list(range(config.num_layers))
        config.reverse_e_h_norm = True
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4MtpWeight


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
register_model("deepseek_v4_mtp", DeepSeekV4Mtp, ["DeepseekV4ForCausalLMNextN"])
