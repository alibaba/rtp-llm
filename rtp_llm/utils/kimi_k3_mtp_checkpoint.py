"""Validate the standalone K3 MTP checkpoint without loading tensor payloads.

Actual weight loading still uses the K3 fastsafetensors/MXFP4 loader. This
preflight checks the source shapes before TP/EP transformations can hide errors.
"""

import argparse
import json
import math
import struct
from pathlib import Path

IGNORED_SUFFIXES = (
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
)


def mtp_source_layer(text):
    """vLLM starts MTP at num_hidden_layers, independently of shard layout."""
    source = text.get("num_hidden_layers")
    if type(source) is not int or source < 1:
        raise ValueError("K3 MTP requires a positive num_hidden_layers")
    if text.get("num_nextn_predict_layers") != 1:
        raise ValueError("K3 MTP currently supports one recurrent nextn layer")
    return source


def mtp_layer_prefix(text):
    return f"language_model.model.layers.{mtp_source_layer(text)}."


def expected_tensors(text):
    prefix = mtp_layer_prefix(text)
    h, vocab = text["hidden_size"], text["vocab_size"]
    heads, q, kv = (
        text["num_attention_heads"],
        text["q_lora_rank"],
        text["kv_lora_rank"],
    )
    nope, rope, value = (
        text["qk_nope_head_dim"],
        text["qk_rope_head_dim"],
        text["v_head_dim"],
    )
    latent, inter = text["routed_expert_hidden_size"], text["moe_intermediate_size"]
    shared = text["num_shared_experts"] * inter
    shapes = {
        name: (h,)
        for name in (
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "shared_head.norm.weight",
        )
    }
    shapes.update(
        {
            "embed_tokens.weight": (vocab, h),
            "shared_head.head.weight": (vocab, h),
            "eh_proj.weight": (h, 2 * h),
            "self_attn.q_a_proj.weight": (q, h),
            "self_attn.q_a_layernorm.weight": (q,),
            "self_attn.q_b_proj.weight": (heads * (nope + rope), q),
            "self_attn.kv_a_proj_with_mqa.weight": (kv + rope, h),
            "self_attn.kv_a_layernorm.weight": (kv,),
            "self_attn.kv_b_proj.weight": (heads * (nope + value), kv),
            "self_attn.g_proj.weight": (heads * value, h),
            "self_attn.o_proj.weight": (h, heads * value),
            "block_sparse_moe.gate.weight": (text["num_experts"], h),
            "block_sparse_moe.routed_expert_down_proj.weight": (latent, h),
            "block_sparse_moe.routed_expert_norm.weight": (latent,),
            "block_sparse_moe.routed_expert_up_proj.weight": (h, latent),
            "block_sparse_moe.shared_experts.gate_proj.weight": (shared, h),
            "block_sparse_moe.shared_experts.up_proj.weight": (shared, h),
            "block_sparse_moe.shared_experts.down_proj.weight": (h, shared),
        }
    )
    result = {prefix + key: (shape, "BF16") for key, shape in shapes.items()}
    result[prefix + "block_sparse_moe.gate.e_score_correction_bias"] = (
        (text["num_experts"],),
        "F32",
    )
    if latent % 32 or inter % 32:
        raise ValueError("K3 MTP MXFP4 dimensions must be divisible by group size 32")
    for expert in range(text["num_experts"]):
        for projection, rows, cols in (
            ("w1", inter, latent),
            ("w2", latent, inter),
            ("w3", inter, latent),
        ):
            name = prefix + f"block_sparse_moe.experts.{expert}.{projection}."
            result[name + "weight_packed"] = ((rows, cols // 2), "U8")
            result[name + "weight_scale"] = ((rows, cols // 32), "U8")
    return result


def validate_checkpoint(checkpoint, config=None):
    root = Path(checkpoint)
    if config is None:
        config = json.loads((root / "config.json").read_text())
    text = config["text_config"]
    prefix = mtp_layer_prefix(text)
    ignored = {prefix + suffix for suffix in IGNORED_SUFFIXES}
    quant = text["quantization_config"]
    groups = quant["config_groups"].values()
    if (
        quant["format"] != "mxfp4-pack-quantized"
        or not quant["config_groups"]
        or any(
            group["weights"]["group_size"] != 32
            or group["weights"]["num_bits"] != 4
            for group in groups
        )
    ):
        raise ValueError("K3 MTP requires checkpoint-native group-32 MXFP4")
    expected = expected_tensors(text)
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    missing = expected.keys() - weight_map.keys()
    unexpected = weight_map.keys() - expected.keys() - ignored
    if missing or unexpected:
        raise ValueError(
            f"K3 MTP manifest mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    shards = set(weight_map.values())
    validated = set()
    for shard in sorted(shards):
        path = root / shard
        size = path.stat().st_size
        with path.open("rb") as reader:
            header_size = struct.unpack("<Q", reader.read(8))[0]
            if header_size > min(size - 8, 64 * 1024 * 1024):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(reader.read(header_size))
        indexed = {key for key, source in weight_map.items() if source == shard}
        if set(header) - {"__metadata__"} != indexed:
            raise ValueError(f"K3 MTP shard/index tensor mismatch: {shard}")
        for name in indexed:
            if name in ignored:
                continue
            entry = header[name]
            shape, dtype = expected[name]
            if tuple(entry["shape"]) != shape or entry["dtype"] != dtype:
                raise ValueError(
                    f"K3 MTP tensor mismatch: {name}: expected {shape}/{dtype}, got {entry}"
                )
            begin, end = entry["data_offsets"]
            nbytes = math.prod(shape) * {"U8": 1, "BF16": 2, "F32": 4}[dtype]
            if begin < 0 or end - begin != nbytes or end > size - 8 - header_size:
                raise ValueError(f"K3 MTP truncated or invalid tensor payload: {name}")
            validated.add(name)
    return {
        "shards": len(shards),
        "required_tensors": len(validated),
        "ignored_attnres_tensors": len(set(weight_map) & ignored),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(validate_checkpoint(args.checkpoint), indent=2))
