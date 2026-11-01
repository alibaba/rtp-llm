#!/usr/bin/env python3
"""Deterministic, CPU-only real tokenizer/safetensors Engram fixture.

This is a complete two-layer Engram component checkpoint, not a complete LLM:
there are no attention/MoE/main embedding weights. The owner checkpoint contains
the tables and local projection weights; the consumer has ONLY WKV/q/k weights
and the tokenizer. This makes accidental consumer table loading fail visibly.

Run from the exact RTP main checkout/venv. Imports of torch, transformers and
Engram are lazy; this module never imports native engine ops or allocates CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAYERS = ((0, 4097), (1, 514))
HEAD_DIM = 64
HC = 2
HIDDEN_DIM = 128
MAX_NGRAM = 4
HEADS = 2
WKV_K = (MAX_NGRAM - 1) * HEADS * HEAD_DIM
WKV_N = (HC + 1) * HIDDEN_DIM
MODEL_REVISION = "engram-framework-fixture-main-492ecf45-v2"
IMAGE_TOKEN_ID = 4
_ROW_ID_BITS = (max(rows for _, rows in LAYERS) - 1).bit_length()
_ROW_TAG_BITS = _ROW_ID_BITS + (len(LAYERS) - 1).bit_length()


def _json(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    path.chmod(0o600)


def _sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    tokens = ["<pad>", "<unk>", "<bos>", "<eos>", "<image>"]
    tokens += [f"word{index:02d}" for index in range(27)]
    backend = Tokenizer(WordLevel({token: index for index, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>",
        bos_token="<bos>", eos_token="<eos>", additional_special_tokens=["<image>"],
        model_max_length=128,
    )
    if len(tokenizer) != 32 or tokenizer.convert_tokens_to_ids("<image>") != IMAGE_TOKEN_ID:
        raise RuntimeError("fixture tokenizer identity changed")
    return tokenizer


def table_rows(layer_id, row_ids):
    """Produce selected FP8/E8M0 rows with a unique layer/row sign signature."""
    import torch

    layer_slots = {layer: slot for slot, (layer, _) in enumerate(LAYERS)}
    if layer_id not in layer_slots:
        raise ValueError("unknown fixture layer")
    row_ids = torch.as_tensor(row_ids, dtype=torch.int64, device="cpu")
    if bool(((row_ids < 0) | (row_ids >= dict(LAYERS)[layer_id])).any()):
        raise ValueError("fixture row index is outside its layer")
    if _ROW_TAG_BITS > HEAD_DIM:
        raise ValueError("fixture row signature does not fit the embedding width")
    columns = torch.arange(HEAD_DIM, dtype=torch.int64)
    groups = torch.arange(HEAD_DIM // 32, dtype=torch.int64)
    values = ((row_ids[..., None] * 7 + columns * 3 + layer_id * 11) % 17 - 8).float() / 4
    # Positive E8M0 scales preserve these signs after dequantization. Thus all
    # actual rows, including rows from different layers, remain distinguishable
    # without increasing the numeric range or losing exact FP8 representation.
    row_tag = row_ids | (layer_slots[layer_id] << _ROW_ID_BITS)
    bits = (row_tag[..., None] >> torch.arange(_ROW_TAG_BITS, dtype=torch.int64)) & 1
    values[..., :_ROW_TAG_BITS] = bits.float() * 2 - 1
    weights = values.to(torch.float8_e4m3fn).contiguous()
    scale_bytes = ((row_ids[..., None] * 2 + groups + layer_id) % 4 + 125).to(torch.uint8)
    scales = scale_bytes.contiguous().view(torch.float8_e8m0fnu)
    return weights, scales


def local_weights(layer_id):
    """Deterministic small native-format WKV and BF16 gate projections."""
    import torch

    rows = torch.arange(WKV_N, dtype=torch.int64)[:, None]
    columns = torch.arange(WKV_K, dtype=torch.int64)[None, :]
    weight = (((rows * 13 + columns * 7 + layer_id * 19) % 17 - 8).float() / 64).to(torch.float8_e4m3fn)
    scale_rows = torch.arange(WKV_N // 32, dtype=torch.int64)[:, None]
    scale_columns = torch.arange(WKV_K // 32, dtype=torch.int64)[None, :]
    scales = ((scale_rows + scale_columns + layer_id) % 3 + 125).to(torch.uint8).view(torch.float8_e8m0fnu)
    gate = torch.arange(HC * HIDDEN_DIM, dtype=torch.int64).reshape(HC, HIDDEN_DIM)
    return {
        "wkv.weight": weight.contiguous(),
        "wkv.scale": scales.contiguous(),
        "q_weight": (1 + ((gate + layer_id) % 13 - 6).float() / 32).to(torch.bfloat16),
        "k_weight": (1 + ((gate * 3 + layer_id) % 17 - 8).float() / 32).to(torch.bfloat16),
    }


def make_inputs(batch_size, step=0):
    """Reproducible CPU inputs: explicit history, boundaries and masked images.

    Use these tensors independently for each consumer; no owner tensor or full
    table is required. ``step`` changes decode history and hidden activations.
    """
    import torch

    if type(batch_size) is not int or batch_size < 0 or type(step) is not int or step < 0:
        raise ValueError("batch_size and step must be nonnegative integers")
    rows = torch.arange(batch_size, dtype=torch.int64)
    history = torch.arange(MAX_NGRAM, dtype=torch.int64)
    positions = rows[:, None] * 3 + step - history[None, :]
    windows = torch.where(positions >= 0, 5 + positions * 5 % 27, -1).to(torch.int32)
    # A singleton is one real rolling decode sequence: at steps 0/1/7 it has
    # 0/1/3 history tokens. Multi-row cases additionally inject independent
    # sequence boundaries, current image tokens and fully masked rows.
    if batch_size > 1:
        windows[rows % 3 == 0, 1:] = -1
        windows[rows % 5 == 4, 0] = IMAGE_TOKEN_ID
        windows[rows % 7 == 6, :] = -1
    hidden_values = torch.arange(batch_size * HC * HIDDEN_DIM, dtype=torch.int64)
    hidden = (((hidden_values + step * 13) % 31 - 15).float() / 16).to(torch.bfloat16)
    hidden = hidden.reshape(batch_size, HC, HIDDEN_DIM)
    return {
        "hidden": hidden,
        "engram_token_windows": windows,
        "token_mask": (windows[:, 0] >= 0) & (windows[:, 0] != IMAGE_TOKEN_ID),
    }


def _save_checkpoint(path, tokenizer, config, *, include_tables):
    import torch
    from safetensors.torch import save_file

    path.mkdir(mode=0o700)
    tokenizer.save_pretrained(str(path))
    _json(path / "config.json", {**config, "checkpoint_path": str(path)})
    weight_map, tensor_bytes = {}, 0
    for layer_id, row_count in LAYERS:
        prefix = f"layers.{layer_id}.engram."
        local = {prefix + name: tensor for name, tensor in local_weights(layer_id).items()}
        filename = f"layer-{layer_id:02d}-local.safetensors"
        save_file(local, str(path / filename), metadata={"format": "pt"})
        for name, tensor in local.items():
            weight_map[name] = filename
            tensor_bytes += tensor.numel() * tensor.element_size()
        if include_tables:
            weights, scales = table_rows(layer_id, torch.arange(row_count))
            tables = {prefix + "embed.weight": weights, prefix + "embed.scale": scales}
            filename = f"layer-{layer_id:02d}-embedding.safetensors"
            save_file(tables, str(path / filename), metadata={"format": "pt"})
            for name, tensor in tables.items():
                weight_map[name] = filename
                tensor_bytes += tensor.numel() * tensor.element_size()
    _json(path / "model.safetensors.index.json", {
        "metadata": {"total_size": tensor_bytes}, "weight_map": weight_map,
    })
    for file in path.iterdir():
        if file.is_file():
            file.chmod(0o600)
    if not include_tables and any(".embed." in name for name in weight_map):
        raise AssertionError("consumer checkpoint unexpectedly includes embedding weights")
    return weight_map


def generate_fixture(root: Path) -> dict:
    """Create a fresh fixture and return config pointing to consumer_checkpoint.

    Both checkpoints have independently reloadable real tokenizer files. The
    owner directory contains all twelve Engram tensors. Consumer contains only
    eight local tensors, with no embedding entries or files. Reusing an existing
    destination is rejected rather than overwriting another experiment.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from rtp_llm.models_py.modules.dsv41.engram import (
        EngramLayout, NgramHashState, build_compressed_token_map,
    )
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import hash_state_digest

    root = Path(root)
    if not root.is_absolute():
        raise ValueError("fixture root must be absolute")
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    tokenizer = _tokenizer()
    _, compressed_size = build_compressed_token_map(tokenizer)
    config = dict(
        model_type="deepseek_v41", fixture_component_only=True,
        engram_layer_ids=[layer for layer, _ in LAYERS],
        engram_num_embeddings=[rows for _, rows in LAYERS],
        engram_max_ngram_size=MAX_NGRAM, engram_n_heads=HEADS,
        engram_head_dim=HEAD_DIM, engram_compressed_vocab_size=compressed_size,
        engram_pad_token_id=tokenizer.pad_token_id, engram_vocab_size=17,
        image_token_id=IMAGE_TOKEN_ID, rms_norm_eps=1e-6,
        hidden_size=HIDDEN_DIM, hc_mult=HC, num_hidden_layers=len(LAYERS),
        vocab_size=len(tokenizer), max_position_embeddings=128,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        dtype="bfloat16", checkpoint_path=str(root / "consumer_checkpoint"),
    )
    layout = EngramLayout(config)
    if layout.n_hash_cols * layout.head_dim != WKV_K:
        raise AssertionError("real Engram hash geometry differs from WKV fixture")
    owner_path, consumer_path = root / "owner_checkpoint", root / "consumer_checkpoint"
    owner_map = _save_checkpoint(owner_path, tokenizer, config, include_tables=True)
    consumer_map = _save_checkpoint(consumer_path, tokenizer, config, include_tables=False)
    if len(owner_map) != 12 or len(consumer_map) != 8:
        raise AssertionError("fixture checkpoint tensor coverage is incomplete")
    # Use the actual from_pretrained route used by the model, not an injected map.
    owner_hash = NgramHashState(layout, str(owner_path))
    consumer_hash = NgramHashState(layout, str(consumer_path))
    hash_abi = hash_state_digest(owner_hash)
    if hash_state_digest(consumer_hash) != hash_abi:
        raise AssertionError("owner/consumer saved tokenizer hash semantics differ")

    inputs, cases = {}, []
    for label, batch, step in (("decode0", 1, 0), ("decode1", 1, 1),
                               ("decode7", 1, 7), ("mixed8", 8, 2),
                               ("prefill33", 33, 3), ("empty", 0, 0)):
        case = make_inputs(batch, step)
        case["hash_ids"] = consumer_hash.hash_token_windows(
            case["engram_token_windows"],
            dead_mask=case["engram_token_windows"] == IMAGE_TOKEN_ID,
        )
        inputs.update({f"{label}.{name}": tensor.contiguous() for name, tensor in case.items()})
        cases.append(dict(name=label, batch_size=batch, step=step))
    save_file(inputs, str(root / "oracle_inputs.safetensors"), metadata={"format": "pt"})
    (root / "oracle_inputs.safetensors").chmod(0o600)
    _json(root / "config.json", config)

    # Verify actual serialized dtype/shape and all required keys on the CPU.
    expected_dtypes = {"embed.weight": torch.float8_e4m3fn, "embed.scale": torch.float8_e8m0fnu,
                       "wkv.weight": torch.float8_e4m3fn, "wkv.scale": torch.float8_e8m0fnu,
                       "q_weight": torch.bfloat16, "k_weight": torch.bfloat16}
    for checkpoint, mapping in ((owner_path, owner_map), (consumer_path, consumer_map)):
        for name, filename in mapping.items():
            suffix = name.split(".engram.", 1)[1]
            layer_id = int(name.split(".")[1])
            row_count = dict(LAYERS)[layer_id]
            expected_shape = {
                "embed.weight": (row_count, HEAD_DIM),
                "embed.scale": (row_count, HEAD_DIM // 32),
                "wkv.weight": (WKV_N, WKV_K), "wkv.scale": (WKV_N // 32, WKV_K // 32),
                "q_weight": (HC, HIDDEN_DIM), "k_weight": (HC, HIDDEN_DIM),
            }[suffix]
            with safe_open(checkpoint / filename, framework="pt", device="cpu") as file:
                tensor = file.get_tensor(name)
            if (tensor.device.type != "cpu" or tensor.dtype != expected_dtypes[suffix]
                    or tuple(tensor.shape) != expected_shape):
                raise AssertionError(f"unexpected serialized fixture dtype/shape for {suffix}")
    files = {str(path.relative_to(root)): _sha256(path.read_bytes())
             for path in sorted(root.rglob("*")) if path.is_file()}
    _json(root / "fixture.json", {
        "schema_version": 1, "fixture_component_only": True,
        "model_revision": MODEL_REVISION, "hash_abi": hash_abi,
        "owner_checkpoint": "owner_checkpoint", "consumer_checkpoint": "consumer_checkpoint",
        "oracle_inputs": "oracle_inputs.safetensors", "cases": cases,
        "layers": [list(pair) for pair in LAYERS], "head_dim": HEAD_DIM,
        "hc": HC, "hidden_dim": HIDDEN_DIM, "wkv_k": WKV_K, "wkv_n": WKV_N,
        "owner_tensor_count": len(owner_map), "consumer_tensor_count": len(consumer_map),
        "consumer_has_embedding_files": False, "files_sha256": files,
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="new absolute destination")
    args = parser.parse_args()
    config = generate_fixture(args.root)
    metadata = json.loads((args.root / "fixture.json").read_text())
    print(json.dumps({"created": str(args.root), "checkpoint_path": config["checkpoint_path"],
                      "hash_abi": metadata["hash_abi"], "owner_tensors": metadata["owner_tensor_count"],
                      "consumer_tensors": metadata["consumer_tensor_count"],
                      "cases": [case["name"] for case in metadata["cases"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
