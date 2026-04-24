import functools
import json
import logging
import os
from typing import List

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2, DeepSeekV2Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    transpose,
    yarn_get_mscale,
    zeros,
)


class DeepSeekV4Weight(DeepSeekV2Weight):
    """V4 weight loader.

    The V4 release on HuggingFace uses a flat checkpoint layout (`embed.weight`,
    `layers.{i}.attn.wq_a.weight`, ...) rather than the `model.layers.{i}.self_attn.*`
    convention used by V2/V3. Override the weight key map accordingly.

    NOTE: The full V4 weight set (mHC params, compressor, indexer, grouped o-proj,
    attn_sink, hash router) is not enumerated yet — see TODO below. M0 wires only
    the keys required to confirm config parsing + registration end-to-end.
    """

    # TODO(deepseek-v4): enumerate all V4-specific weight tensors:
    # - layers.{i}.hc_attn_fn / hc_attn_base / hc_attn_scale (mHC residual)
    # - layers.{i}.hc_ffn_fn / hc_ffn_base / hc_ffn_scale
    # - layers.{i}.attn.attn_sink (per-head learnable sink logit)
    # - layers.{i}.attn.wq_a / wq_b / q_norm (Q LoRA)
    # - layers.{i}.attn.wkv / kv_norm (single-MQA-head KV)
    # - layers.{i}.attn.wo_a / wo_b (grouped output projection, n_groups=8)
    # - layers.{i}.attn.compressor.* (only for compress_ratio in {4,128})
    # - layers.{i}.attn.indexer.* (only for compress_ratio == 4)
    # - layers.{i}.ffn.gate.weight / .bias (or .tid2eid for hash layers)
    # - layers.{i}.ffn.experts.{e}.w1/w2/w3 (FP4 dtype for V4-Flash)
    # - layers.{i}.ffn.shared_experts.w1/w2/w3
    # - hc_head_fn / hc_head_base / hc_head_scale (top-level HC head)


class DeepSeekV4MtpWeight(DeepSeekV4Weight):
    """V4 MTP block: Block + e_proj/h_proj/enorm/hnorm + own ParallelHead.

    Single nextN block. Layer index in checkpoint is `args.n_layers + 0` for the first
    (and only) MTP layer; the released V4-Flash has 1 MTP layer at logical index 43.
    """


class DeepSeekV4(DeepSeekV2):
    """DeepSeek-V4 top-level model.

    Inherits config defaults from DeepSeekV2; overrides _from_hf to parse V4-specific
    fields (CSA/HCA per-layer dispatch, mHC residual, grouped output projection,
    attention sink, sliding-window bypass, sqrt(softplus) scoring, hash routing,
    SwiGLU clamp, partial RoPE with separate compress base).

    NOTE: _create_python_model is intentionally left as the V2 implementation for
    M0; the V2 GenericMoeModel will not work for V4 (CSA/HCA + mHC are completely
    different from MLA). M2/M3 must replace this with a dedicated V4 compute graph.
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
        config.max_seq_len = 4096
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        DeepSeekV4._from_hf(config, ckpt_path)
        return config

    def _create_python_model(self):
        # M0: stub — V2's MLA-based GenericMoeModel cannot handle V4's CSA/HCA + mHC.
        # M2/M3 will provide rtp_llm.models_py.model_desc.deepseek_v4_model:DeepSeekV4Model.
        raise NotImplementedError(
            "DeepSeek-V4 Python compute graph is not implemented yet. "
            "M0 wires config parsing + registration only; M2/M3 will provide the "
            "CSA/HCA attention modules and mHC residual."
        )

    @staticmethod
    def _from_hf(config: ModelConfig, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        # ---- Basic shape ----
        # V4 uses MQA: kv_head_num = 1, single head_dim = 512 (no separate qk_nope/v_head).
        # The Q LoRA path (q_lora_rank=1024) is followed by a single grouped output proj.
        config.num_layers = config_json["num_hidden_layers"]
        config.hidden_size = config_json["hidden_size"]
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-6)
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config_json.get("num_key_value_heads", 1)
        # In V4, head_dim is the *full* Q head dim (e.g. 512). RoPE applies only to the
        # last `qk_rope_head_dim` (e.g. 64) — partial RoPE.
        head_dim = config_json["head_dim"]
        rope_head_dim = config_json["qk_rope_head_dim"]
        config.attn_config.size_per_head = head_dim
        # Reuse MLA dim slots to carry V4 head decomposition through the engine.
        # V4 has no kv_lora_rank (true MQA), but we still set q_lora_rank for the Q-LoRA path.
        config.attn_config.use_mla = False
        config.attn_config.q_lora_rank = config_json.get("q_lora_rank", 0)
        config.attn_config.kv_lora_rank = 0
        config.attn_config.nope_head_dim = head_dim - rope_head_dim
        config.attn_config.rope_head_dim = rope_head_dim
        config.attn_config.v_head_dim = head_dim

        # ---- V4 hybrid attention (CSA + HCA + SWA bypass) ----
        config.attn_config.use_v4_hybrid_attn = True
        layer_compress_ratios = config_json.get("compress_ratios")
        if layer_compress_ratios is None:
            raise ValueError(
                "DeepSeek-V4 config.json missing required field `compress_ratios`."
            )
        # Length must be num_hidden_layers (+1 for MTP if present).
        n_mtp = config_json.get("num_nextn_predict_layers", 0)
        expected = config.num_layers + n_mtp
        if len(layer_compress_ratios) != expected:
            raise ValueError(
                f"compress_ratios length {len(layer_compress_ratios)} != "
                f"num_hidden_layers ({config.num_layers}) + num_nextn ({n_mtp})"
            )
        config.attn_config.layer_compress_ratios = layer_compress_ratios
        config.attn_config.sliding_window = int(config_json.get("sliding_window", 0))
        config.attn_config.has_attention_sink = True
        config.attn_config.o_groups = int(config_json.get("o_groups", 0))
        config.attn_config.o_lora_rank = int(config_json.get("o_lora_rank", 0))
        config.attn_config.compress_rope_theta = int(
            config_json.get("compress_rope_theta", 0)
        )

        # ---- mHC (Manifold-Constrained Hyper-Connections) residual ----
        config.attn_config.hc_mult = int(config_json.get("hc_mult", 0))
        config.attn_config.hc_sinkhorn_iters = int(
            config_json.get("hc_sinkhorn_iters", 20)
        )
        config.attn_config.hc_eps = float(config_json.get("hc_eps", 1e-6))

        # ---- RoPE (yarn, partial-RoPE, separate base for compressed branch) ----
        config.attn_config.rope_config.base = int(
            config_json.get("rope_theta", config.attn_config.rope_config.base)
        )
        config.attn_config.rope_config.dim = rope_head_dim
        config.attn_config.rope_config.offset = head_dim - rope_head_dim
        config.attn_config.rope_config.style = 0
        rope_scaling = config_json.get("rope_scaling")
        if rope_scaling is not None:
            scaling_factor = rope_scaling["factor"]
            config.attn_config.rope_config.scale = scaling_factor
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            # V4 omits explicit mscale fields → use scaling_factor-derived defaults
            mscale = rope_scaling.get("mscale", 1.0)
            mscale_all_dim = rope_scaling.get("mscale_all_dim", 1.0)
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            if mscale != 1.0 or mscale_all_dim != 1.0:
                config.attn_config.rope_config.mscale = yarn_get_mscale(
                    scaling_factor, mscale
                ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
                softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                config.attn_config.softmax_extra_scale = (
                    softmax_mscale * softmax_mscale
                )

        # ---- MoE ----
        scoring_func = config_json.get("scoring_func", "softmax")
        if scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func == "sigmoid":
            config.scoring_func = 1
        elif scoring_func == "sqrtsoftplus":
            config.scoring_func = 2
        else:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

        config.routed_scaling_factor = float(
            config_json.get("routed_scaling_factor", 1.0)
        )
        config.moe_k = int(config_json["num_experts_per_tok"])
        config.expert_num = int(config_json["n_routed_experts"])
        moe_intermediate_size = int(config_json["moe_intermediate_size"])
        # V4 drops node-grouped routing constraint — n_group / topk_group absent.
        config.moe_n_group = 1
        config.moe_topk_group = 1
        n_shared_experts = int(config_json.get("n_shared_experts", 1))
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = bool(config_json.get("norm_topk_prob", True))
        # All-MoE: every layer uses MoE FFN (V4 has no dense FFN replacement).
        config.moe_style = 1
        config.moe_layer_index = list(range(config.num_layers))
        config.moe_hash_routing_layers = int(config_json.get("num_hash_layers", 0))
        config.swiglu_limit = float(config_json.get("swiglu_limit", 0.0))

        config.config_dtype = config_json.get("torch_dtype", None)
        logging.info(
            "DeepSeek-V4 config parsed: layers=%d hidden=%d head_num=%d head_dim=%d "
            "rope_dim=%d q_lora=%d o_groups=%d o_lora=%d hc_mult=%d sliding=%d "
            "experts=%d/%d hash_layers=%d scoring=%s swiglu_limit=%.1f",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            head_dim,
            rope_head_dim,
            config.attn_config.q_lora_rank,
            config.attn_config.o_groups,
            config.attn_config.o_lora_rank,
            config.attn_config.hc_mult,
            config.attn_config.sliding_window,
            config.moe_k,
            config.expert_num,
            config.moe_hash_routing_layers,
            scoring_func,
            config.swiglu_limit,
        )

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4Weight


class DeepSeekV4Mtp(DeepSeekV4):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # MTP block lives at the last entry of compress_ratios (which is 0 = SWA-only for V4).
        config.moe_layer_index = list(range(config.num_layers))
        config.reverse_e_h_norm = True
        config.is_mtp = True
        return config

    @staticmethod
    def get_weight_cls():
        return DeepSeekV4MtpWeight


register_model("deepseek_v4", DeepSeekV4, ["DeepseekV4ForCausalLM"])
register_model("deepseek_v4_mtp", DeepSeekV4Mtp, ["DeepseekV4ForCausalLMNextN"])
