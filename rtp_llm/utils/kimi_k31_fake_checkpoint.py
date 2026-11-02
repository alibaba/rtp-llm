"""Stream deterministic, real-shape K3.1 fake weights to a separate 3FS fixture.

Tensor templates come from pinned checkpoint headers. Synthetic nextn weights
exercise the existing recurrent MTP interface and are not native K3.1 weights.
"""

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from rtp_llm.utils.kimi_k3_mtp_checkpoint import expected_tensors

STORAGE_ROOT = Path("/mnt/hf3fs/3fs/models/kimi/Kimi-K3.1")


def fixture_config(source, layers, seed, *, mtp=False):
    if layers not in (4, 93):
        raise ValueError("target fixtures use four or 93 layers")
    config = copy.deepcopy(source)
    text = config["text_config"]
    if source.get("model_type") != "kimi_k31" or text["num_hidden_layers"] != 93:
        raise ValueError("fixture source must be the complete K3.1 config")
    text["num_hidden_layers"] = layers
    text["num_nextn_predict_layers"] = int(mtp)
    linear = text["linear_attn_config"]
    for key in ("kda_layers", "full_attn_layers"):
        linear[key] = [i for i in linear[key] if i <= layers]
    text["dsa_config"]["dsa_layers"] = list(linear["full_attn_layers"])
    if mtp:
        linear["full_attn_layers"].append(layers + 1)
        text["dsa_config"]["dsa_layers"].append(layers + 1)
    config["rtp_llm_fixture"] = {
        "kind": "synthetic_dsa_mtp" if mtp else "fake_target",
        "seed": seed, "target_layers": layers,
        "accuracy_scope": "structure, distributed equivalence and performance only",
        "native_k31_mtp": False,
    }
    return config


def target_tensors(config, templates):
    source = templates["tensors"]
    text = config["text_config"]
    for name, spec in source.items():
        if ".layers." not in name:
            yield name, spec
    dsa_layers = set(text["dsa_config"]["dsa_layers"])
    dense_count = text["first_k_dense_replace"]
    for layer in range(text["num_hidden_layers"]):
        attention_source = 3 if layer + 1 in dsa_layers else 0
        attention_prefix = f"language_model.model.layers.{attention_source}."
        dense = layer < dense_count or layer % text.get("moe_layer_freq", 1) != 0
        ffn_prefix = "language_model.model.layers.0." if dense else "language_model.model.layers.1."
        destination = f"language_model.model.layers.{layer}."
        for name, spec in source.items():
            if name.startswith(attention_prefix):
                suffix = name[len(attention_prefix):]
                if suffix.startswith("self_attn.") or suffix in (
                    "input_layernorm.weight", "post_attention_layernorm.weight",
                    "self_attention_res_norm.weight", "self_attention_res_proj.weight",
                    "mlp_res_norm.weight", "mlp_res_proj.weight",
                ):
                    yield destination + suffix, spec
            if name.startswith(ffn_prefix):
                suffix = name[len(ffn_prefix):]
                if not suffix.startswith(("mlp.", "block_sparse_moe.")):
                    continue
                if "experts.0." in suffix:
                    for expert in range(text["num_experts"]):
                        yield destination + suffix.replace("experts.0.", f"experts.{expert}."), spec
                else:
                    yield destination + suffix, spec


def _canonical_seed_name(name):
    if name.endswith("embed_tokens.weight"):
        return "language_model.model.embed_tokens.weight"
    if name.endswith("shared_head.head.weight"):
        return "language_model.lm_head.weight"
    return name


def _tensor_chunks(name, spec, seed, chunk_elems=4 * 1024 * 1024):
    import numpy as np

    digest = hashlib.sha256(f"{seed}:{_canonical_seed_name(name)}".encode()).digest()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "little")))
    count = math.prod(spec["shape"])
    dtype = spec["dtype"]
    for offset in range(0, count, chunk_elems):
        size = min(chunk_elems, count - offset)
        if dtype == "U8":
            if name.endswith("weight_scale"):
                # Finite UE8M0 2^-10 scales; these are checkpoint MXFP4 scales.
                values = np.full(size, 117, dtype=np.uint8)
            elif name.endswith("weight_packed"):
                # Signed finite E2M1 values, independently selected per nibble.
                low = rng.integers(0, 5, size=size, dtype=np.uint8)
                high = rng.integers(0, 5, size=size, dtype=np.uint8)
                low |= rng.integers(0, 2, size=size, dtype=np.uint8) << 3
                high |= rng.integers(0, 2, size=size, dtype=np.uint8) << 3
                values = low | (high << 4)
            else:
                raise ValueError(f"unsupported byte tensor: {name}")
            yield values.tobytes()
            continue
        if dtype not in ("BF16", "F32"):
            raise ValueError(f"unsupported fake tensor dtype: {dtype}")
        if name.endswith("A_log") or name.endswith("dt_bias") or name.endswith(".bias") or name.endswith("e_score_correction_bias"):
            values = np.zeros(size, dtype=np.float32)
        elif "norm" in name or "layernorm" in name:
            values = np.ones(size, dtype=np.float32)
        else:
            fan_in = spec["shape"][-1] if len(spec["shape"]) > 1 else spec["shape"][0]
            amplitude = 0.05 / math.sqrt(fan_in)
            values = rng.uniform(-amplitude, amplitude, size=size).astype(np.float32)
        if dtype == "F32":
            yield values.astype("<f4", copy=False).tobytes()
        else:
            bits = values.view(np.uint32)
            # Round finite FP32 to BF16, ties to even, without a GPU dependency.
            rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            yield rounded.astype("<u2").tobytes()


def _write_shard(root, kind, number, count, group, seed):
    filename = f"model-{number:05d}-of-{count:06d}.safetensors"
    header = {"__metadata__": {"format": "pt", "fixture": kind}}
    cursor = 0
    for name, spec, nbytes in group:
        header[name] = {**spec, "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    digest = hashlib.sha256()
    with (root / filename).open("xb") as writer:
        for payload in (struct.pack("<Q", len(encoded)), encoded):
            writer.write(payload)
            digest.update(payload)
        for name, spec, nbytes in group:
            written = 0
            for payload in _tensor_chunks(name, spec, seed):
                writer.write(payload)
                digest.update(payload)
                written += len(payload)
            if written != nbytes:
                raise ValueError(f"fake tensor byte count mismatch: {name}")
    return filename, digest.hexdigest(), cursor


def _write_shards(root, kind, groups, seed, workers):
    # RNG state is per tensor, so worker scheduling cannot change checkpoint bytes.
    # Each worker owns a distinct new file and only bounded NumPy chunks.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_write_shard, root, kind, number, len(groups), group, seed): number
            for number, group in enumerate(groups, 1)
        }
        for future in as_completed(pending):
            number = pending[future]
            result = future.result()
            print(json.dumps({"shard": number, "shards": len(groups),
                              "payload_bytes": result[2]}), flush=True)
            yield result


def write_checkpoint(output, config, specs, seed, *, shard_bytes=1024**3, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("fake checkpoint workers must be between 1 and 8")
    output = Path(output)
    root = STORAGE_ROOT.resolve(strict=True)
    resolved = output.resolve()
    protected = Path("/3fs-data/3fs/mtp_test").resolve()
    if root not in resolved.parents or resolved == protected or protected in resolved.parents:
        raise ValueError("fake weights require a new checkpoint below the independent K3.1 3FS root")
    fs = subprocess.check_output(["findmnt", "-T", str(root), "-n", "-o", "FSTYPE"], text=True).strip()
    if fs != "fuse.hf3fs":
        raise ValueError("fake checkpoint writer requires the verified 3FS mount")
    resolved.mkdir()  # Never overwrite or resume an existing checkpoint implicitly.
    (resolved / "INCOMPLETE").write_text("generation in progress\n")
    ordered = sorted(specs.items())
    groups, group, size = [], [], 0
    item_bytes = {"U8": 1, "BF16": 2, "F32": 4}
    for name, spec in ordered:
        nbytes = math.prod(spec["shape"]) * item_bytes[spec["dtype"]]
        if group and size + nbytes > shard_bytes:
            groups.append(group)
            group, size = [], 0
        group.append((name, spec, nbytes))
        size += nbytes
    if group:
        groups.append(group)
    weight_map, shard_hashes, total_size = {}, {}, 0
    for number, group in enumerate(groups, 1):
        filename = f"model-{number:05d}-of-{len(groups):06d}.safetensors"
        for name, _, _ in group:
            weight_map[name] = filename
    for filename, digest, cursor in _write_shards(
        resolved, config["rtp_llm_fixture"]["kind"], groups, seed, workers
    ):
        total_size += cursor
        shard_hashes[filename] = digest
    (resolved / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (resolved / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n")
    (resolved / "fixture.json").write_text(json.dumps({
        **config["rtp_llm_fixture"], "shards": dict(sorted(shard_hashes.items())),
        "tensor_count": len(specs), "payload_bytes": total_size,
        "generator": "numpy PCG64; per-tensor SHA256-derived seed; bounded chunks",
        "load_method": "fastsafetensors",
        "generation_workers": workers,
    }, indent=2) + "\n")
    (resolved / "INCOMPLETE").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tensor-specs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", type=int, choices=(4, 93), default=4)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--synthetic-mtp", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    source = json.loads(Path(args.config).read_text())
    templates = json.loads(Path(args.tensor_specs).read_text())
    config = fixture_config(source, args.layers, args.seed, mtp=args.synthetic_mtp)
    config["rtp_llm_fixture"]["source_revision"] = templates["revision"]
    if args.synthetic_mtp:
        specs = {name: {"shape": list(shape), "dtype": dtype}
                 for name, (shape, dtype) in expected_tensors(config["text_config"]).items()}
    else:
        entries = list(target_tensors(config, templates))
        specs = dict(entries)
        if len(entries) != len(specs):
            raise ValueError("duplicate fake target tensor names")
    write_checkpoint(args.output, config, specs, args.seed, workers=args.workers)


if __name__ == "__main__":
    main()
