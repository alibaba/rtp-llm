"""CPU-only schema audit for released MiniMax-M3.1 DSpARK checkpoints.

This validates metadata, not forward mathematics or numerical quality. It uses
only the standard library and reads safetensors headers, never tensor payloads.
Run directly with ``python minimax_m31_dspark_checkpoint.py CHECKPOINT/dspark``.
"""

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict

PREFIX = "language_model.model.dspark."
UNVERIFIED_MATH = [
    "exact target hidden capture positions",
    "confidence STS temperature calibration and adaptive scheduling policy",
    "numerical agreement with the training/reference forward implementation",
]


def read_safetensors_header(path: Path) -> Dict[str, Any]:
    """Read and check tensor extents without reading any payload bytes."""
    with path.open("rb") as reader:
        prefix = reader.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size > 100_000_000 or size > path.stat().st_size - 8:
            raise ValueError(f"{path}: invalid safetensors header length {size}")
        header = json.loads(reader.read(size))
    header.pop("__metadata__", None)
    payload_size = path.stat().st_size - 8 - size
    dtype_bytes = {"BF16": 2, "F8_E4M3": 1, "U8": 1}
    end = 0
    for name, tensor in sorted(
        header.items(), key=lambda item: item[1]["data_offsets"][0]
    ):
        shape, dtype = tensor["shape"], tensor["dtype"]
        if not isinstance(shape, list) or any(
            type(v) is not int or v < 0 for v in shape
        ):
            raise ValueError(f"{name}: invalid shape {shape}")
        if dtype not in dtype_bytes:
            raise ValueError(f"{name}: unsupported checkpoint dtype {dtype}")
        start, stop = tensor["data_offsets"]
        if start != end or stop - start != math.prod(shape) * dtype_bytes[dtype]:
            raise ValueError(f"{name}: invalid or non-contiguous tensor extent")
        end = stop
    if end != payload_size:
        raise ValueError(
            f"{path}: tensor extents end at {end}, payload has {payload_size} bytes"
        )
    return header


def validate_schema(config: Dict[str, Any], tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the released five-layer SWA/dense schema; reject mock overlays."""
    text = config.get("text_config", config)

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    require(
        config.get("architectures") == ["DSparkMiniMaxDraftModel"],
        "expected DSparkMiniMaxDraftModel architecture",
    )
    for key, expected in {
        "num_hidden_layers": 5,
        "hidden_size": 6144,
        "vocab_size": 200064,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "dense_intermediate_size": 12288,
        "dspark_ffn_hidden_size": 12288,
        "dspark_markov_rank": 256,
        "dspark_block_size": 7,
        "dspark_noise_token_id": 200058,
        "sliding_window": 4096,
        "dspark_markov_head_type": "vanilla",
        "dspark_use_dense_ffn": True,
        "dspark_hybrid_context_fusion": False,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "use_gemma_norm": True,
    }.items():
        require(
            text.get(key) == expected,
            f"text_config.{key}: expected {expected!r}, got {text.get(key)!r}",
        )
        if key in config:
            require(
                config[key] == expected, f"top-level {key} conflicts with text_config"
            )
    target_layers = [3, 17, 31, 45, 59]
    require(
        text.get("dspark_target_layer_ids") == target_layers,
        "unexpected target hidden layer ids",
    )
    require(
        text.get("layer_types") == ["sliding_attention"] * 5,
        "all five draft layers must use sliding_attention",
    )
    require(text.get("moe_layer_freq") == [0] * 5, "draft must use dense FFN, not MoE")
    require(
        text.get("sparse_attention_config") is None, "draft must not use sparse MSA"
    )
    require(
        text.get("dspark_config", {}).get("hybrid_context_fusion") is False,
        "hybrid context fusion is unsupported",
    )
    quant = config.get("quantization_config", {})
    require(
        quant.get("quant_method") == "mxfp8"
        and quant.get("weight_block_size") == [1, 32],
        "expected MXFP8 weights with [1, 32] scales",
    )

    expected = {}

    def tensor(name: str, dtype: str, shape: list) -> None:
        expected[PREFIX + name] = (dtype, shape)

    tensor("fc.weight", "BF16", [6144, 30720])
    tensor("hidden_norm.weight", "BF16", [6144])
    tensor("final_norm.weight", "BF16", [6144])
    for suffix in ("markov_w1", "markov_w2"):
        tensor(f"markov_head.{suffix}.weight", "BF16", [200064, 256])
    tensor("confidence_head.proj.weight", "BF16", [1, 6400])
    tensor("confidence_head.proj.bias", "BF16", [1])
    for layer in range(5):
        root = f"layers.{layer}.decoder_layer."
        for name in ("input_layernorm", "post_attention_layernorm"):
            tensor(root + name + ".weight", "BF16", [6144])
        for name in ("q_norm", "k_norm"):
            tensor(root + "self_attn." + name + ".weight", "BF16", [128])
        for name, rows, cols in (
            ("self_attn.q_proj", 8192, 6144),
            ("self_attn.k_proj", 512, 6144),
            ("self_attn.v_proj", 512, 6144),
            ("self_attn.o_proj", 6144, 8192),
            ("mlp.gate_proj", 12288, 6144),
            ("mlp.up_proj", 12288, 6144),
            ("mlp.down_proj", 6144, 12288),
        ):
            tensor(root + name + ".weight", "F8_E4M3", [rows, cols])
            tensor(root + name + ".weight_scale_inv", "U8", [rows, cols // 32])
    missing, extra = sorted(expected.keys() - tensors.keys()), sorted(
        tensors.keys() - expected.keys()
    )
    require(
        not missing and not extra,
        f"tensor name mismatch: missing={missing}, unexpected={extra}",
    )
    for name, (dtype, shape) in expected.items():
        actual = tensors[name]
        require(
            actual.get("dtype") == dtype and actual.get("shape") == shape,
            f"{name}: expected {dtype} {shape}, got {actual.get('dtype')} {actual.get('shape')}",
        )
    return {
        "schema": "minimax_m31_dspark_preview2",
        "tensor_count": len(tensors),
        "draft_layers": 5,
        "attention": "sliding_attention",
        "block_size": 7,
        "sliding_window": 4096,
        "layer_types": ["sliding_attention"] * 5,
        "use_gemma_norm": True,
        "ffn": "dense_mxfp8",
        "target_layer_ids": target_layers,
        "aux_feature_dim": 30720,
        "shared_embedding_and_lm_head_absent": True,
        "required_shared_weights": [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
        ],
        "unverified_math": list(UNVERIFIED_MATH),
        "runtime_ready": False,
    }


def inspect_checkpoint(checkpoint: str) -> Dict[str, Any]:
    """Validate config, indexed tensor names, shard extents, and total byte size."""
    root = Path(checkpoint)
    config = json.loads((root / "config.json").read_text())
    if config.get("architectures") != ["DSparkMiniMaxDraftModel"]:
        nested = root / "dspark"
        if (nested / "config.json").is_file():
            raise ValueError(
                f"{root}: target checkpoint root was passed as SP_CHECKPOINT_PATH; "
                f"use the nested DSpARK checkpoint {nested}"
            )
        raise ValueError(
            f"{root}: expected DSparkMiniMaxDraftModel checkpoint, got "
            f"architectures={config.get('architectures')!r}"
        )
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors = {}
    total_bytes = 0
    for shard in sorted(set(weight_map.values())):
        if Path(shard).name != shard:
            raise ValueError(f"expected a local shard basename, got {shard!r}")
        header = read_safetensors_header(root / shard)
        if set(header) != {
            name for name, filename in weight_map.items() if filename == shard
        }:
            raise ValueError(f"{shard}: index and safetensors header disagree")
        tensors.update(header)
        total_bytes += sum(
            t["data_offsets"][1] - t["data_offsets"][0] for t in header.values()
        )
    if index.get("metadata", {}).get("total_size") != total_bytes:
        raise ValueError("index total_size does not match tensor payload bytes")
    report = validate_schema(config, tensors)
    report.update(
        checkpoint=str(root.resolve()),
        tensor_payload_bytes=total_bytes,
        shards=sorted(set(weight_map.values())),
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2))
