"""Checkpoint inventory for V4.1; header validation never materializes weights."""

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rtp_llm.config.dsv41_config import V41Config


@dataclass(frozen=True)
class V41TensorSpec:
    name: str
    shape: tuple[int, ...] | None
    dtype: str | None
    placement: str = "gpu"
    conversion: str = "identity"


def build_v41_manifest(config: V41Config) -> dict[str, V41TensorSpec]:
    t, v = config.text, config.vision
    dim, head, rank = t["hidden_size"], t["head_dim"], t["q_lora_rank"]
    hc, groups, o_rank = t["hc_mult"], t["o_groups"], t["o_lora_rank"]
    specs: dict[str, V41TensorSpec] = {}

    def add(name, shape, dtype="BF16", placement="gpu", conversion="identity"):
        if name in specs:
            raise ValueError(f"duplicate weight descriptor: {name}")
        specs[name] = V41TensorSpec(
            name,
            tuple(shape) if shape is not None else None,
            dtype,
            placement,
            conversion,
        )

    def dense(name, shape, conversion="identity"):
        add(name + ".weight", shape, "F8_E4M3", conversion=conversion)
        add(
            name + ".scale",
            ((shape[0] + 31) // 32, (shape[1] + 31) // 32),
            "F8_E8M0",
            conversion=conversion,
        )

    for family, layers, experts in (
        ("layers", t["num_hidden_layers"], t["n_routed_experts"]),
        ("mtp", t["num_nextn_predict_layers"], t["dspark_n_routed_experts"]),
    ):
        for layer in range(layers):
            p = f"{family}.{layer}."
            for norm in ("attn_norm", "ffn_norm"):
                add(p + norm + ".weight", (dim,))
            for part in ("attn", "ffn"):
                add(p + f"hc_{part}_fn", ((2 + hc) * hc, hc * dim), "F32")
                add(p + f"hc_{part}_base", ((2 + hc) * hc,), "F32")
                add(p + f"hc_{part}_scale", (3,), "F32")
            add(p + "attn.attn_sink", (t["num_attention_heads"],), "F32")
            add(p + "attn.q_norm.weight", (rank,))
            add(p + "attn.kv_norm.weight", (head,))
            for name, shape in (
                ("wq_a", (rank, dim)),
                ("wq_b", (t["num_attention_heads"] * head, rank)),
                ("wkv", (head, dim)),
                ("wo_b", (dim, groups * o_rank)),
            ):
                dense(p + "attn." + name, shape)
            dense(
                p + "attn.wo_a",
                (groups * o_rank, t["num_attention_heads"] * head // groups),
                "dequantize_bf16",
            )
            if family == "layers" and layer in config.kv_owners:
                add(p + "attn.compressor.wkv.weight", (head, dim))
                add(p + "attn.compressor.norm.weight", (head,))
                if t["compress_ratios"][layer] == 2:
                    add(p + "attn.compressor.wgate.weight", (head, dim))
                add(p + "attn.indexer.wk.weight", (t["index_head_dim"], head))
                add(p + "attn.indexer.k_norm.weight", (t["index_head_dim"],))
            if family == "layers" and layer in config.index_owners:
                dense(
                    p + "attn.indexer.wq_b",
                    (t["index_n_heads"] * t["index_head_dim"], rank),
                )
                add(p + "attn.indexer.weights_proj.weight", (t["index_n_heads"], dim))
            add(p + "ffn.gate.weight", (experts, dim))
            add(p + "ffn.gate.bias", (experts,), "F32")
            add(p + "ffn.gate.bias_vl", (experts,), "F32")
            inter = t["moe_intermediate_size"]
            for expert in range(experts):
                for name, shape in (
                    ("w1", (inter, dim)),
                    ("w2", (dim, inter)),
                    ("w3", (inter, dim)),
                ):
                    base = p + f"ffn.experts.{expert}.{name}"
                    add(base + ".weight", (shape[0], shape[1] // 2), "I8")
                    add(base + ".scale", (shape[0], shape[1] // 32), "F8_E8M0")
            for name, shape in (
                ("w1", (inter * t["n_shared_experts"], dim)),
                ("w2", (dim, inter * t["n_shared_experts"])),
                ("w3", (inter * t["n_shared_experts"], dim)),
            ):
                dense(p + "ffn.shared_experts." + name, shape)
    for layer, rows in zip(t["engram_layer_ids"], t["engram_num_embeddings"]):
        p = f"layers.{layer}.engram."
        e_dim = t["engram_head_dim"]
        add(p + "embed.weight", (rows, e_dim), "F8_E4M3", "host_shared")
        add(p + "embed.scale", (rows, e_dim // 32), "F8_E8M0", "host_shared")
        dense(
            p + "wkv",
            (
                dim * (hc + 1),
                (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"] * e_dim,
            ),
        )
        add(p + "q_weight", (hc, dim))
        add(p + "k_weight", (hc, dim))
    add("embed.weight", (t["vocab_size"], dim))
    add("head.weight", (t["vocab_size"], dim), conversion="fp32_logits")
    add("norm.weight", (dim,))
    dense("mtp.0.main_proj", (dim, dim * len(t["dspark_target_layer_ids"])))
    add("mtp.0.main_norm.weight", (dim,))
    last = t["num_nextn_predict_layers"] - 1
    add(f"mtp.{last}.norm.weight", (dim,))
    add(
        f"mtp.{last}.markov_head.embed.weight",
        (t["vocab_size"], t["dspark_markov_rank"]),
    )
    add(
        f"mtp.{last}.markov_head.head.weight",
        (t["vocab_size"], t["dspark_markov_rank"]),
    )
    add(f"mtp.{last}.confidence_head.proj.weight", (1, dim + t["dspark_markov_rank"]))
    for name in ("image_start", "image_end", "image_newline"):
        add(name, (dim,))
    add("vision.patch_embed.proj.weight", (v["hidden_size"], 3 * v["patch_size"] ** 2))
    add("vision.patch_embed.proj.bias", (v["hidden_size"],))
    add("vision.norm.weight", (v["hidden_size"],), conversion="fp32_norm")
    vd, vi = v["hidden_size"], v["intermediate_size"]
    for layer in range(v["num_hidden_layers"]):
        p = f"vision.blocks.{layer}."
        for name in ("norm1", "norm2"):
            add(p + name + ".weight", (vd,), conversion="fp32_norm")
        for name, shape in (("attn.wqkv", (3 * vd, vd)), ("attn.wo", (vd, vd))):
            add(p + name + ".weight", shape)
            add(p + name + ".bias", (shape[0],))
        add(p + "mlp.w1.weight", (2 * vi, vd))
        add(p + "mlp.w2.weight", (vd, vi))
    add("aligner.w1.weight", (dim, vd * v["downsample_ratio"] ** 2))
    add("aligner.w1.bias", (dim,))
    add("aligner.w2.weight", (dim, dim))
    add("aligner.w2.bias", (dim,))
    return specs


def validate_inventory(
    specs: dict[str, V41TensorSpec], keys: Iterable[str], *, include_draft: bool = True
) -> dict[str, list[str]]:
    keys = set(keys)
    ignored = {key for key in specs if not include_draft and key.startswith("mtp.")}
    required = set(specs) - ignored
    missing = required - keys
    unexpected = keys - set(specs)
    if missing or unexpected:
        raise ValueError(
            f"V4.1 weight inventory mismatch: missing={sorted(missing)[:20]}, unexpected={sorted(unexpected)[:20]}"
        )
    return {"required": sorted(required), "ignored": sorted(keys & ignored)}


def validate_checkpoint_headers(checkpoint: str | Path, config: V41Config) -> dict:
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    raw_index = index_path.read_bytes()
    index = json.loads(raw_index)
    specs = build_v41_manifest(config)
    validate_inventory(specs, index["weight_map"])
    observed = set()
    reports = []
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("checkpoint index must reference local shard basenames")
        path = checkpoint / shard
        with path.open("rb") as reader:
            size = reader.read(8)
            if len(size) != 8:
                raise ValueError(f"truncated safetensors length in {shard}")
            header_length = struct.unpack("<Q", size)[0]
            if not 2 <= header_length <= 64 * 1024 * 1024:
                raise ValueError(f"invalid safetensors header length in {shard}")
            raw_header = reader.read(header_length)
            if len(raw_header) != header_length:
                raise ValueError(f"truncated safetensors header in {shard}")
        header = json.loads(raw_header)
        regions = []
        payload_bytes = path.stat().st_size - 8 - header_length
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in observed or index["weight_map"].get(name) != shard:
                raise ValueError(f"duplicate or misassigned tensor {name}")
            spec = specs[name]
            if spec.shape is not None and tuple(item["shape"]) != spec.shape:
                raise ValueError(f"{name}: shape {item['shape']} != {spec.shape}")
            if spec.dtype is not None and item["dtype"] != spec.dtype:
                raise ValueError(f"{name}: dtype {item['dtype']} != {spec.dtype}")
            begin, end = item["data_offsets"]
            if (
                type(begin) is not int
                or type(end) is not int
                or not 0 <= begin <= end <= payload_bytes
            ):
                raise ValueError(f"{name}: tensor payload is truncated")
            dtype_bytes = {"BF16": 2, "F32": 4, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1}[
                item["dtype"]
            ]
            if end - begin != math.prod(item["shape"]) * dtype_bytes:
                raise ValueError(f"{name}: tensor extent does not match shape/dtype")
            regions.append((begin, end, name))
            observed.add(name)
        previous_end = 0
        for begin, end, name in sorted(regions):
            if begin != previous_end:
                raise ValueError(f"{name}: overlapping or non-contiguous tensor data")
            previous_end = end
        if previous_end != payload_bytes:
            raise ValueError(f"{shard}: unindexed trailing tensor data")
        reports.append(
            {
                "shard": shard,
                "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                "file_bytes": path.stat().st_size,
            }
        )
    validate_inventory(specs, observed)
    return {
        "tensor_count": len(observed),
        "shard_count": len(reports),
        "index_sha256": hashlib.sha256(raw_index).hexdigest(),
        "shards": reports,
        "deferred_shape_validation": [
            name for name, spec in specs.items() if spec.shape is None
        ],
        "payload_checksum_verified": False,
    }
