#!/usr/bin/env python3
"""Build deterministic DFlash2 (DFlashQwenDraftModel) mock checkpoints for Qwen4.5-Flash."""

import argparse
from collections import OrderedDict
import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

from safetensors import safe_open
from safetensors.torch import save_file
import torch


DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "synthetic-8l": {
        "base_ckpt": "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-8L-qsa-mock",
        "output_dir": "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-8L-dflash-mock",
        "num_hidden_layers": 5,
        "hidden_size": 512,
        "intermediate_size": 128,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "target_layer_ids": [0, 1, 2, 3, 4],
        "num_target_layers": 8,
        "block_size": 8,
        "gamma": 8,
        "mask_token_id": 3,
        "sample_from_anchor": True,
        "hc_count": 4,
        "hc_lowrank": 32,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 64,
        "selector_top_k": 16,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "routed_expert_hidden_size": 256,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 128,
        "tie_word_embeddings": False,
    },
    "40l": {
        "base_ckpt": "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-1109-qsa-mock",
        "output_dir": "/data0/luohaocheng.lhc/models/Qwen4.5-Flash-1109-dflash-mock",
        "num_hidden_layers": 5,
        "hidden_size": 5120,
        "intermediate_size": 1280,
        "num_attention_heads": 32,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "target_layer_ids": [0, 5, 10, 15, 20],
        "num_target_layers": 40,
        "block_size": 8,
        "gamma": 8,
        "mask_token_id": 248280,
        "sample_from_anchor": True,
        "hc_count": 4,
        "hc_lowrank": 128,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "num_experts": 640,
        "num_experts_per_tok": 10,
        "routed_expert_hidden_size": 2560,
        "moe_intermediate_size": 2560,
        "shared_expert_intermediate_size": 1280,
        "tie_word_embeddings": False,
    },
}

TOKENIZER_FILES: Tuple[str, ...] = (
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "tokenizer.model",
    "added_tokens.json",
)

DTYPE_BYTES: Dict[str, int] = {"bf16": 2, "fp32": 4, "u8": 1}


def _modules_to_not_convert(num_layers: int) -> List[str]:
    modules: List[str] = [
        "candidate_selector.hidden_projection",
        "candidate_selector.predecessor_codebook",
        "candidate_selector.successor_codebook",
        "fc",
        "hc_mixer.input_mix_weight_down",
        "hc_mixer.input_mix_weight_up",
    ]
    for i in range(num_layers):
        modules.extend(
            [
                f"layers.{i}.attention_conv.base_kernel",
                f"layers.{i}.attention_conv.kernel_projection",
                f"layers.{i}.attn_hc.block_inject_weight",
                f"layers.{i}.attn_hc.input_mix_weight_down",
                f"layers.{i}.attn_hc.input_mix_weight_up",
                f"layers.{i}.mlp.expert_bias",
                f"layers.{i}.mlp.gate",
                f"layers.{i}.mlp.latent_down_proj",
                f"layers.{i}.mlp.latent_gate_proj",
                f"layers.{i}.mlp.latent_scale_gate",
                f"layers.{i}.mlp.latent_up_proj",
                f"layers.{i}.mlp.shared_expert.down_proj",
                f"layers.{i}.mlp.shared_expert.gate_proj",
                f"layers.{i}.mlp.shared_expert.up_proj",
                f"layers.{i}.mlp.shared_expert_gate",
                f"layers.{i}.mlp_conv.base_kernel",
                f"layers.{i}.mlp_conv.kernel_projection",
                f"layers.{i}.mlp_hc.block_inject_weight",
                f"layers.{i}.mlp_hc.input_mix_weight_down",
                f"layers.{i}.mlp_hc.input_mix_weight_up",
                f"layers.{i}.self_attn.k_proj",
                f"layers.{i}.self_attn.o_proj",
                f"layers.{i}.self_attn.output_gate_proj",
                f"layers.{i}.self_attn.q_proj",
                f"layers.{i}.self_attn.v_proj",
            ]
        )
    return modules


def build_dflash2_config(
    base_cfg: Dict[str, Any], profile: Dict[str, Any], *, seed: int = 42
) -> Dict[str, Any]:
    del seed
    num_layers = int(profile["num_hidden_layers"])
    hidden_size = int(profile["hidden_size"])
    vocab_size = int(base_cfg["vocab_size"])
    head_num = int(profile["num_attention_heads"])
    kv_head_num = int(profile["num_key_value_heads"])
    head_dim = int(profile["head_dim"])
    num_target_layers = int(base_cfg.get("num_hidden_layers", profile["num_target_layers"]))
    target_layer_ids = [int(x) for x in profile["target_layer_ids"]]
    mask_token_id = int(profile["mask_token_id"])
    if mask_token_id >= vocab_size:
        mask_token_id = min(3, vocab_size - 1)

    layer_types = ["sliding_attention"] * max(0, num_layers - 1) + ["full_attention"]
    full_ids = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]

    rope_params = dict(base_cfg.get("rope_parameters") or {})
    partial_rotary = float(
        base_cfg.get("partial_rotary_factor", rope_params.get("partial_rotary_factor", 0.25))
    )
    rope_theta = float(base_cfg.get("rope_theta", rope_params.get("rope_theta", 10000.0)))
    rms_norm_eps = float(base_cfg.get("rms_norm_eps", 1e-6))
    max_pos = int(base_cfg.get("max_position_embeddings", 8192))

    return {
        "architectures": ["DFlashQwenDraftModel"],
        "model_type": "qwen3",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "situ_glu",
        "initializer_range": 0.02,
        "tie_word_embeddings": bool(profile["tie_word_embeddings"]),
        "use_cache": False,
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "hidden_size": hidden_size,
        "intermediate_size": int(profile["intermediate_size"]),
        "num_hidden_layers": num_layers,
        "num_attention_heads": head_num,
        "num_key_value_heads": kv_head_num,
        "head_dim": head_dim,
        "rms_norm_eps": rms_norm_eps,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_pos,
        "rope_theta": rope_theta,
        "rope_scaling": None,
        "partial_rotary_factor": partial_rotary,
        "rotary_percent": partial_rotary,
        "layer_types": layer_types,
        "sliding_window": 2049,
        "use_sliding_window": True,
        "max_window_layers": max(0, num_layers - 1),
        "is_causal": False,
        "block_size": int(profile["block_size"]),
        "sample_from_anchor": bool(profile["sample_from_anchor"]),
        "gamma": int(profile["gamma"]),
        "num_target_layers": num_target_layers,
        "dflash_config": {
            "block_size": int(profile["block_size"]),
            "sample_from_anchor": bool(profile["sample_from_anchor"]),
            "mask_token_id": mask_token_id,
            "mask_mode": "consecutive",
            "is_causal": False,
            "swa_window_size": 2048,
            "full_attention_layer_ids": full_ids,
            "num_target_layers": num_target_layers,
            "target_layer_ids": target_layer_ids,
            "target_hidden_source": "post_next_hc_mix",
            "target_hidden_size": hidden_size,
            "target_feature_width_per_layer": hidden_size,
            "fc_in_features": len(target_layer_ids) * hidden_size,
            "target_uses_hyper_connection": True,
            "target_hc_count": int(profile["hc_count"]),
            "target_uses_ple": True,
            "kv_inject_method": "fc",
            "use_pre_fc_norm": False,
            "pre_fc_norm_units": 0,
            "qk_layernorm": True,
            "use_post_mlp_layernorm": False,
            "gated_norms": [],
            "gated_layernorm_lowrank": 0,
            "zero_centered_gamma": {
                "block_norms": True,
                "hidden_norm": True,
                "pre_fc_norm": True,
                "hc_norm": True,
            },
            "use_attn_output_gate": True,
            "rotary_percent": partial_rotary,
            "partial_rotary_factor": partial_rotary,
            "rope_is_neox_style": True,
            "use_hc": True,
            "hc_has_parameters": True,
            "hc_count": int(profile["hc_count"]),
            "hc_lowrank": int(profile["hc_lowrank"]),
            "hc_method": "gated_residual_simple",
            "hc_final_method": "gated_residual_simple",
            "hc_per_branch_norm": True,
            "use_final_hc_mixer": True,
            "has_final_layernorm": False,
            "moe_router_pre_softmax": True,
            "moe_norm_topk_scores": True,
            "conv_kernel_size": int(profile["conv_kernel_size"]),
            "conv_group_size": int(profile["conv_group_size"]),
            "selector_rank": int(profile["selector_rank"]),
            "selector_top_k": int(profile["selector_top_k"]),
            "conv_base_kernel_identity_folded": True,
            "converter": "convert_dflash_mcore_to_hf_v2.py",
        },
        "num_experts": int(profile["num_experts"]),
        "num_experts_per_tok": int(profile["num_experts_per_tok"]),
        "moe_intermediate_size": int(profile["moe_intermediate_size"]),
        "shared_expert_intermediate_size": int(profile["shared_expert_intermediate_size"]),
        "use_shared_expert_gate": True,
        "norm_topk_prob": True,
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": _modules_to_not_convert(num_layers),
        },
        "hidden_act_gate_beta": 4.0,
        "hidden_act_up_beta": 25.0,
        "clip_mlp_output": 1000.0,
        "first_k_dense_replace": 0,
        "first_k_moe_replace": 0,
        "routed_expert_hidden_size": int(profile["routed_expert_hidden_size"]),
        "router_score_function": "sigmoid",
        "router_precision": "fp32_logits",
        "router_normalize_topk_scores": True,
        "moe_router_pre_softmax": True,
        "moe_router_enable_expert_bias": True,
        "shared_expert_gate": True,
        "hc_count": int(profile["hc_count"]),
        "hc_lowrank": int(profile["hc_lowrank"]),
        "hc_per_branch_norm": True,
        "hc_rmsnorm_bias": True,
        "hc_prev_mix": True,
        "use_ple": False,
    }


def _tensor_specs(cfg: Dict[str, Any]) -> List[Tuple[str, Tuple[int, ...], str, str]]:
    """Return ``(key, shape, dtype_tag, init_kind)`` matching DFlashQwenDraftModel."""
    dflash = cfg["dflash_config"]
    h = int(cfg["hidden_size"])
    v = int(cfg["vocab_size"])
    L = int(cfg["num_hidden_layers"])
    n_h = int(cfg["num_attention_heads"])
    n_kv = int(cfg["num_key_value_heads"])
    d = int(cfg["head_dim"])
    hc = int(cfg["hc_count"])
    h_hc = hc * h
    hc_lr = int(cfg["hc_lowrank"])
    rank = int(dflash["selector_rank"])
    taps = int(dflash["conv_kernel_size"])
    gsize = int(dflash["conv_group_size"])
    conv_proj_out = 2 * taps * (h // gsize)
    fc_in = int(dflash["fc_in_features"])

    num_experts = int(cfg["num_experts"])
    r_h = int(cfg["routed_expert_hidden_size"])
    m_i = int(cfg["moe_intermediate_size"])
    s_i = int(cfg["shared_expert_intermediate_size"])

    specs: List[Tuple[str, Tuple[int, ...], str, str]] = [
        ("model.embed_tokens.weight", (v, h), "bf16", "randn"),
        ("fc.weight", (h, fc_in), "bf16", "randn"),
        ("hidden_norm.weight", (h,), "bf16", "zeros"),
        ("candidate_selector.hidden_projection.weight", (rank, h), "bf16", "randn"),
        ("candidate_selector.predecessor_codebook", (v, rank), "bf16", "randn"),
        ("candidate_selector.successor_codebook", (v, rank), "bf16", "randn"),
        ("hc_mixer.hc_norm.weight", (h_hc,), "bf16", "zeros"),
        ("hc_mixer.hc_norm.bias", (h_hc,), "bf16", "zeros"),
        ("hc_mixer.input_mix_weight_down.weight", (hc_lr, h_hc), "bf16", "randn"),
        ("hc_mixer.input_mix_weight_up.weight", (h_hc, hc_lr), "bf16", "randn"),
    ]
    if not cfg.get("tie_word_embeddings", False):
        specs.append(("lm_head.weight", (v, h), "bf16", "randn"))

    for i in range(L):
        p = f"layers.{i}."
        up_out = h if i == 0 else h_hc
        specs.extend(
            [
                (f"{p}attn_hc.hc_norm.weight", (h_hc,), "bf16", "zeros"),
                (f"{p}attn_hc.hc_norm.bias", (h_hc,), "bf16", "zeros"),
                (f"{p}attn_hc.block_inject_weight.weight", (hc, h_hc), "bf16", "randn"),
                (f"{p}attn_hc.input_mix_weight_down.weight", (hc_lr, h_hc), "bf16", "randn"),
                (f"{p}attn_hc.input_mix_weight_up.weight", (up_out, hc_lr), "bf16", "randn"),
                (f"{p}mlp_hc.hc_norm.weight", (h_hc,), "bf16", "zeros"),
                (f"{p}mlp_hc.hc_norm.bias", (h_hc,), "bf16", "zeros"),
                (f"{p}mlp_hc.block_inject_weight.weight", (hc, h_hc), "bf16", "randn"),
                (f"{p}mlp_hc.input_mix_weight_up.weight", (h_hc, hc_lr), "bf16", "randn"),
            ]
        )
        if i + 1 < L:
            specs.append(
                (f"{p}mlp_hc.input_mix_weight_down.weight", (hc_lr, h_hc), "bf16", "randn")
            )
        specs.extend(
            [
                (f"{p}attention_conv.base_kernel", (2, taps, h), "bf16", "conv_identity"),
                (f"{p}attention_conv.kernel_projection.weight", (conv_proj_out, h), "bf16", "randn"),
                (f"{p}mlp_conv.base_kernel", (2, taps, h), "bf16", "conv_identity"),
                (f"{p}mlp_conv.kernel_projection.weight", (conv_proj_out, h), "bf16", "randn"),
                (f"{p}self_attn.q_proj.weight", (n_h * d, h), "bf16", "randn"),
                (f"{p}self_attn.k_proj.weight", (n_kv * d, h), "bf16", "randn"),
                (f"{p}self_attn.v_proj.weight", (n_kv * d, h), "bf16", "randn"),
                (f"{p}self_attn.o_proj.weight", (h, n_h * d), "bf16", "randn"),
                (f"{p}self_attn.output_gate_proj.weight", (n_h * d, h), "bf16", "randn"),
                (f"{p}self_attn.q_norm.weight", (d,), "bf16", "zeros"),
                (f"{p}self_attn.k_norm.weight", (d,), "bf16", "zeros"),
                (f"{p}mlp.latent_down_proj", (r_h, h), "bf16", "randn"),
                (f"{p}mlp.latent_gate_proj", (r_h, h), "bf16", "randn"),
                (f"{p}mlp.latent_up_proj", (h, r_h), "bf16", "randn"),
                (f"{p}mlp.latent_scale_gate", (1, h), "bf16", "randn"),
                (f"{p}mlp.pre_moe_latent_layernorm.weight", (r_h,), "bf16", "zeros"),
                (f"{p}mlp.gate.weight", (num_experts, r_h), "bf16", "randn"),
                (f"{p}mlp.expert_bias", (num_experts,), "fp32", "zeros"),
                (f"{p}mlp.shared_expert.gate_proj.weight", (s_i, h), "bf16", "randn"),
                (f"{p}mlp.shared_expert.up_proj.weight", (s_i, h), "bf16", "randn"),
                (f"{p}mlp.shared_expert.down_proj.weight", (h, s_i), "bf16", "randn"),
                (f"{p}mlp.shared_expert_gate.weight", (1, h), "bf16", "randn"),
            ]
        )
        for e in range(num_experts):
            ep = f"{p}mlp.experts.{e}."
            specs.extend(
                [
                    (f"{ep}gate_proj.weight", (m_i, r_h // 2), "u8", "u8_data"),
                    (f"{ep}gate_proj.weight_scale", (m_i, r_h // 32), "u8", "u8_scale"),
                    (f"{ep}up_proj.weight", (m_i, r_h // 2), "u8", "u8_data"),
                    (f"{ep}up_proj.weight_scale", (m_i, r_h // 32), "u8", "u8_scale"),
                    (f"{ep}down_proj.weight", (r_h, m_i // 2), "u8", "u8_data"),
                    (f"{ep}down_proj.weight_scale", (r_h, m_i // 32), "u8", "u8_scale"),
                ]
            )
    return specs


def _make_tensor(
    gen: torch.Generator, shape: Tuple[int, ...], dtype_tag: str, init_kind: str
) -> torch.Tensor:
    if dtype_tag == "u8":
        if init_kind == "u8_scale":
            return torch.randint(120, 129, shape, generator=gen, dtype=torch.uint8)
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)
    torch_dtype = torch.float32 if dtype_tag == "fp32" else torch.bfloat16
    if init_kind == "zeros":
        return torch.zeros(shape, dtype=torch_dtype)
    if init_kind == "ones":
        return torch.ones(shape, dtype=torch_dtype)
    if init_kind == "conv_identity":
        t = torch.zeros(shape, dtype=torch_dtype)
        t[:, 0, :] = 1.0
        return t
    if init_kind == "randn":
        return (torch.randn(shape, generator=gen, dtype=torch.float32) * 0.02).to(
            torch_dtype
        )
    raise ValueError(f"Unknown init_kind={init_kind!r}")


def _shard_plans(
    specs: Sequence[Tuple[str, Tuple[int, ...], str, str]],
    max_shard_bytes: int,
) -> List[List[Tuple[str, Tuple[int, ...], str, str]]]:
    shards: List[List[Tuple[str, Tuple[int, ...], str, str]]] = []
    cur: List[Tuple[str, Tuple[int, ...], str, str]] = []
    cur_bytes = 0
    for item in specs:
        _, shape, dtype_tag, _ = item
        nbytes = DTYPE_BYTES[dtype_tag]
        for dim in shape:
            nbytes *= int(dim)
        if cur and cur_bytes + nbytes > max_shard_bytes:
            shards.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(item)
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    return shards


def validate_dflash_checkpoint(output_dir: str) -> Dict[str, Any]:
    """Validate DFlash2 config parsing, safetensors headers, and Qwen4_5DSparkWeight."""
    from rtp_llm.models.qwen4_5.qwen4_5_dspark import Qwen4_5DSpark, Qwen4_5DSparkWeight
    from rtp_llm.utils.model_weight import W

    model_cfg = Qwen4_5DSpark._create_config(output_dir)
    if model_cfg.attn_config.is_causal:
        raise AssertionError("Qwen4_5DSpark attn_config.is_causal must be False")
    if model_cfg.num_layers != 5:
        raise AssertionError(f"Expected 5 draft layers, got {model_cfg.num_layers}")
    if not model_cfg.dspark_target_layer_ids:
        raise AssertionError("dspark_target_layer_ids must be non-empty")
    if not model_cfg.dspark_sample_from_anchor:
        raise AssertionError("DFlash2 requires sample_from_anchor=True")

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        index_json = json.load(f)
    weight_map: Dict[str, str] = index_json["weight_map"]

    header_shapes: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = os.path.join(output_dir, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sl = sf.get_slice(k)
                header_shapes[k] = (tuple(sl.get_shape()), str(sl.get_dtype()))

    missing_in_shards = sorted(set(weight_map.keys()) - set(header_shapes.keys()))
    if missing_in_shards:
        raise AssertionError(f"Index keys missing from shards: {missing_in_shards[:5]}")

    weight_info_builder = Qwen4_5DSparkWeight(model_cfg)
    weight_info_builder._process_meta({}, list(weight_map.keys()))
    weight_info = weight_info_builder._get_weight_info()

    global_names = [w.name for w in weight_info.weights]
    for expected_global in (
        W.embedding,
        W.lm_head,
        W.dspark_fc_w,
        W.dspark_hidden_norm_gamma,
        W.dspark_markov_w1,
        W.dspark_markov_w2,
    ):
        if expected_global not in global_names:
            raise AssertionError(f"Missing global weight mapping: {expected_global}")

    for w in weight_info.weights:
        for ck in w.weights:
            if ck.name not in header_shapes:
                raise AssertionError(f"Missing global tensor {ck.name} in shards")

    for layer_list in weight_info.layer_weights:
        for w in layer_list:
            for comp in w.get_components():
                for ck in comp.weights:
                    if ck.name not in header_shapes:
                        raise AssertionError(f"Missing layer tensor {ck.name} in shards")

    return {
        "num_layers": int(model_cfg.num_layers),
        "dspark_target_layer_ids": list(model_cfg.dspark_target_layer_ids),
        "dspark_noise_token_id": int(model_cfg.dspark_noise_token_id),
        "dspark_markov_rank": int(model_cfg.dspark_markov_rank),
        "dspark_sample_from_anchor": bool(model_cfg.dspark_sample_from_anchor),
        "gen_num_per_cycle": int(model_cfg.gen_num_per_cycle),
        "rope_dim": int(model_cfg.attn_config.rope_config.dim),
        "num_experts": int(model_cfg.expert_num),
        "num_tensors": len(header_shapes),
        "num_shards": len(set(weight_map.values())),
    }


def generate_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    profile = dict(DEFAULT_CONFIGS[args.mode])
    base_ckpt = args.base_ckpt or profile["base_ckpt"]
    output_dir = args.output_dir or profile["output_dir"]

    with open(os.path.join(base_ckpt, "config.json"), "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    cfg = build_dflash2_config(base_cfg, profile, seed=int(args.seed))
    if os.path.isdir(output_dir):
        for old_name in os.listdir(output_dir):
            if old_name.endswith(".safetensors") or old_name in (
                "config.json",
                "model.safetensors.index.json",
                "mock_dflash_manifest.json",
            ):
                os.remove(os.path.join(output_dir, old_name))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    copied_tokenizers: List[str] = []
    for fname in TOKENIZER_FILES:
        src = os.path.join(base_ckpt, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
            copied_tokenizers.append(fname)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed))

    specs = _tensor_specs(cfg)
    shard_plans = _shard_plans(specs, int(args.max_shard_bytes))
    num_shards = len(shard_plans)
    weight_map: Dict[str, str] = OrderedDict()
    total_tensor_bytes = 0
    total_disk_bytes = 0
    shard_summaries: List[Dict[str, Any]] = []

    for idx, shard_items in enumerate(shard_plans, start=1):
        shard_name = f"model-{idx:05d}-of-{num_shards:05d}.safetensors"
        shard_path = os.path.join(output_dir, shard_name)
        tensors: Dict[str, torch.Tensor] = OrderedDict()
        shard_bytes = 0
        for name, shape, dtype_tag, init_kind in shard_items:
            t = _make_tensor(gen, shape, dtype_tag, init_kind)
            tensors[name] = t
            weight_map[name] = shard_name
            nbytes = int(t.numel() * t.element_size())
            shard_bytes += nbytes
            total_tensor_bytes += nbytes
        save_file(tensors, shard_path, metadata={"format": "pt"})
        disk_bytes = os.path.getsize(shard_path)
        total_disk_bytes += disk_bytes
        shard_summaries.append(
            {
                "shard": shard_name,
                "num_tensors": len(shard_items),
                "tensor_bytes": shard_bytes,
                "disk_bytes": disk_bytes,
            }
        )
        tensors.clear()

    index_obj = {
        "metadata": {"total_size": total_tensor_bytes},
        "weight_map": weight_map,
    }
    with open(
        os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(index_obj, f, indent=2)
        f.write("\n")

    validation = validate_dflash_checkpoint(output_dir) if args.validate else {}
    manifest = {
        "mode": args.mode,
        "architecture": "DFlashQwenDraftModel",
        "base_ckpt": base_ckpt,
        "output_dir": output_dir,
        "seed": int(args.seed),
        "total_tensor_bytes": total_tensor_bytes,
        "total_safetensors_disk_bytes": total_disk_bytes,
        "copied_tokenizer_files": copied_tokenizers,
        "validation": validation,
        "shards": shard_summaries,
    }
    with open(
        os.path.join(output_dir, "mock_dflash_manifest.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=sorted(DEFAULT_CONFIGS.keys()),
        required=True,
        help="Target profile to mirror.",
    )
    parser.add_argument("--base-ckpt", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-shard-bytes", type=int, default=1_900_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest = generate_checkpoint(args)
    print(
        json.dumps(
            {
                "mode": manifest["mode"],
                "output_dir": manifest["output_dir"],
                "total_tensor_bytes": manifest["total_tensor_bytes"],
                "total_safetensors_disk_bytes": manifest["total_safetensors_disk_bytes"],
                "validation": manifest["validation"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
