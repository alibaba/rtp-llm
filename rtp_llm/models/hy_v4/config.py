"""Parse HY V4 architecture and per-layer schedules."""

import json
import logging
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import MlaOpsType


def create_hy4_config(ckpt_path: str) -> ModelConfig:
    cmp_value = os.environ.get("RTP_LLM_GLM5_CMP", "0").strip().lower()
    if cmp_value in ("1", "true", "yes", "on"):
        raise ValueError(
            "RTP_LLM_GLM5_CMP is GLM-5-specific and cannot be enabled for HY V4"
        )
    if cmp_value not in ("0", "false", "no", "off", ""):
        raise ValueError(f"invalid RTP_LLM_GLM5_CMP={cmp_value!r}")
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {ckpt_path}")
    with open(config_path) as reader:
        raw = json.load(reader)

    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.model_type = "hy_v4"
    config.norm_type = "rmsnorm"
    config.has_pre_decoder_layernorm = False
    config.has_post_decoder_layernorm = True
    config.activation_type = "SiGLU"

    config.hidden_size = int(raw["hidden_size"])
    config.num_layers = int(raw["num_hidden_layers"])
    config.vocab_size = int(raw["vocab_size"])
    config.max_seq_len = int(raw["max_position_embeddings"])
    config.layernorm_eps = float(raw.get("rms_norm_eps", 1e-6))
    # HY V4 uses a dedicated LayerNorm inside the sparse-attention
    # indexer. Its checkpoint contract fixes epsilon at 1e-6; this is
    # intentionally independent from the model RMSNorm epsilon.
    config.indexer_layernorm_eps = 1e-6
    config.indexer_scale_fmt = "ue8m0"
    # Unlike the shared DSV3.2/GLM5 indexer path, the HY V4 reference
    # quantizes the post-RoPE vectors directly. A Hadamard transform
    # preserves an exact dot product, but changes FP8 absmax scales and
    # can therefore change the selected top-k tokens.
    config.indexer_use_hadamard = False
    config.tie_word_embeddings = bool(raw.get("tie_word_embeddings", False))
    config.enable_fp32_lm_head = bool(raw.get("enable_lm_head_fp32", True))
    config.config_dtype = raw.get("dtype", raw.get("torch_dtype"))

    attn = config.attn_config
    attn.use_mla = True
    attn.head_num = int(raw["num_attention_heads"])
    attn.kv_head_num = int(raw.get("num_key_value_heads", attn.head_num))
    attn.q_lora_rank = int(raw["q_lora_rank"])
    attn.kv_lora_rank = int(raw["kv_lora_rank"])
    attn.nope_head_dim = int(raw["qk_nope_head_dim"])
    attn.rope_head_dim = int(raw["qk_rope_head_dim"])
    attn.v_head_dim = int(raw["v_head_dim"])
    attn.size_per_head = attn.nope_head_dim + attn.rope_head_dim
    attn.rope_config.dim = attn.rope_head_dim
    rope_parameters = raw.get("rope_parameters") or {}
    attn.rope_config.base = float(
        rope_parameters.get("rope_theta", raw.get("rope_theta", 10_000_000))
    )
    attn.rope_config.style = 0 if config.mla_ops_type != MlaOpsType.MHA else 5
    attn.rope_config.offset = attn.nope_head_dim
    attn.rope_config.is_neox_style = not bool(raw.get("rope_interleave", True))
    attn.rope_config.indexer_is_neox_style = not bool(
        raw.get("indexer_rope_interleave", True)
    )

    attn.is_sparse = True
    attn.indexer_head_dim = int(raw["index_head_dim"])
    attn.indexer_head_num = int(raw["index_n_heads"])
    attn.indexer_topk = int(raw["index_topk"])
    config.indexer_types = list(raw.get("indexer_types") or [])
    if not config.indexer_types:
        raise KeyError("HY V4 config must contain indexer_types")
    if len(config.indexer_types) != config.num_layers:
        raise ValueError(
            "HY V4 indexer_types length must equal num_hidden_layers, got "
            f"{len(config.indexer_types)} and {config.num_layers}"
        )
    invalid_indexers = set(config.indexer_types) - {"full", "shared"}
    if invalid_indexers:
        raise ValueError(f"unsupported HY V4 indexer types: {sorted(invalid_indexers)}")
    if config.indexer_types[0] != "full":
        raise ValueError("HY V4 layer 0 indexer must be 'full'")

    config.enable_ihc = bool(raw.get("enable_ihc", True))
    config.hc_mult = int(raw.get("hc_mult", 4))
    config.hc_magnitude = float(raw.get("hc_magnitude", 2.0))
    config.hc_eps = float(raw.get("hc_eps", 1e-6))
    config.gated_mla = bool(raw.get("gated_mla", True))
    config.gating_type = str(raw.get("gating_type", "elementwise"))
    config.learnable_sink = bool(raw.get("learnable_sink", True))
    config.learnable_sink_init = float(raw.get("learnable_sink_init", 0.0))
    if not config.enable_ihc or config.hc_mult != 4:
        raise ValueError("HY V4 backbone requires enable_ihc=true and hc_mult=4")
    if not config.gated_mla or config.gating_type != "elementwise":
        raise ValueError("HY V4 requires elementwise gated MLA")
    if not config.learnable_sink:
        raise ValueError("HY V4 requires learnable attention sinks")
    config.force_sparse_mla = True

    config.dense_inter_size = int(raw["intermediate_size"])
    config.moe_inter_size = int(raw["moe_intermediate_size"])
    n_shared = int(raw.get("n_shared_experts", 1))
    config.inter_size = n_shared * config.moe_inter_size
    config.expert_num = int(raw["n_routed_experts"])
    config.moe_k = int(raw["num_experts_per_tok"])
    config.moe_style = 2
    config.moe_n_group = int(raw.get("n_group", 1))
    config.moe_topk_group = int(raw.get("topk_group", 1))
    scoring = raw.get("scoring_func", "sigmoid")
    if scoring != "sigmoid":
        raise ValueError(f"HY V4 only supports sigmoid routing, got {scoring!r}")
    config.scoring_func = 1
    config.has_moe_norm = bool(raw.get("norm_topk_prob", True))
    config.routed_scaling_factor = float(raw.get("routed_scaling_factor", 2.827))
    config.swiglu_limit = float(raw.get("swiglu_limit", 10.0))
    if config.swiglu_limit <= 0:
        raise ValueError("HY V4 routed experts require a positive swiglu_limit")
    config.mlp_layer_types = list(raw.get("mlp_layer_types") or [])
    if not config.mlp_layer_types:
        raise KeyError("HY V4 config must contain mlp_layer_types")
    if len(config.mlp_layer_types) != config.num_layers:
        raise ValueError(
            "HY V4 mlp_layer_types length must equal num_hidden_layers, got "
            f"{len(config.mlp_layer_types)} and {config.num_layers}"
        )
    invalid_mlp = set(config.mlp_layer_types) - {"dense", "sparse"}
    if invalid_mlp:
        raise ValueError(f"unsupported HY V4 MLP layer types: {invalid_mlp}")
    config.moe_layer_index = [
        i
        for i, layer_type in enumerate(config.mlp_layer_types)
        if layer_type == "sparse"
    ]
    logging.info(
        "HY V4 config loaded: layers=%d hidden=%d heads=%d iHC=%d "
        "experts=%d topk=%d",
        config.num_layers,
        config.hidden_size,
        attn.head_num,
        config.hc_mult,
        config.expert_num,
        attn.indexer_topk,
    )
    return config
