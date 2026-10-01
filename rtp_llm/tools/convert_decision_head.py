#!/usr/bin/env python3
"""Convert external decision-model heads into RTP-LLM's unified "decision" format.

One downstream module (rtp_llm/models/downstream_modules/
decision_module.py) serves every supported decision checkpoint.  The contract
is defined entirely inside the target checkpoint:

  * ``decision_head.safetensors`` -- the head tensors, keyed ``decision_head.*``:
      - linear readout (autojev-27b): ``decision_head.weight``  [num_labels, hidden]
        and optionally ``decision_head.bias``                    [num_labels]
      - pointer head (kev-0.5b):      ``decision_head.q.weight`` [proj, hidden]
        ``decision_head.q.bias`` ``decision_head.k.weight`` ``decision_head.k.bias``
  * ``config.json`` gains a ``"decision"`` block describing the head, the label
    space and the prompt convention, e.g.::

        "decision": {
          "format_version": 1,
          "head_type": "linear" | "pointer",
          "head_weight": "decision_head.weight",           # linear
          "head_bias": "decision_head.bias",               # linear, optional
          "head_query_weight": "decision_head.q.weight",   # pointer
          "head_query_bias": "decision_head.q.bias",
          "head_key_weight": "decision_head.k.weight",
          "head_key_bias": "decision_head.k.bias",
          "num_labels": 255,                               # linear
          "codes": ["A", "B", ...],                        # linear, optional
          "temperature": 2.207...,                         # optional
          "prompt_format": "chat" | "kev_branch",
          "markers": {...},                                # kev_branch
          "marker_token_ids": {"option_end": 151649, "decide": 151661}
        }

  * when the checkpoint uses a ``model.safetensors.index.json`` (sharded HF
    layout) the index ``weight_map`` is extended, because RTP-LLM's loader
    enumerates shard files from the index when one exists.  Unsharded
    checkpoints are discovered by glob and need no index change.

The tool is idempotent: re-running with identical content is a no-op.
Existing differing content is never overwritten without ``--force``.

Examples
--------
# inspect a source head
python convert_decision_head.py inspect --source /models/kev-0.5b/head.pt
python convert_decision_head.py inspect --source /models/autojev-27b

# kev LoRA pointer head -> serving copy of the Qwen2.5-0.5B base checkpoint
python convert_decision_head.py kev \
    --adapter-dir /models/kev-0.5b --target-dir /srv/kev-0.5b-rtp

# autojev-27b linear readout -> autojev-27b checkpoint (in place or a copy)
python convert_decision_head.py autojev \
    --ckpt-dir /models/autojev-27b --target-dir /models/autojev-27b
"""

import argparse
import array
import json
import logging
import math
import os
import struct
import sys
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file, save_file

DECISION_CONFIG_KEY = "decision"
DECISION_HEAD_FILE = "decision_head.safetensors"
DECISION_FORMAT_VERSION = 1

# kev-0.5b reserved markers (kev v0.1.0 kev/model.py SPECIAL, in the logical
# order state, q, opt, /opt, decide).  They reuse existing Qwen2.5 special
# tokens so no embedding rows are added.
KEV_DEFAULT_MARKERS = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "option": "<|box_start|>",
    "option_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}

# Pointer-head key mapping: head.pt key -> safetensors key / config key.
POINTER_TENSORS = {
    "q.weight": ("decision_head.q.weight", "head_query_weight"),
    "q.bias": ("decision_head.q.bias", "head_query_bias"),
    "k.weight": ("decision_head.k.weight", "head_key_weight"),
    "k.bias": ("decision_head.k.bias", "head_key_bias"),
}
LINEAR_WEIGHT_TENSOR = ("decision_head.weight", "head_weight")
LINEAR_BIAS_TENSOR = ("decision_head.bias", "head_bias")


# --------------------------------------------------------------------------- #
# safetensors writing (numpy-free fallback)
# --------------------------------------------------------------------------- #
_ST_DTYPE_NAMES = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}
# unsigned view + array typecode per element size (exact byte passthrough)
_ST_VIEW = {
    1: (torch.uint8, "B"),
    2: (torch.uint16, "H"),
    4: (torch.uint32, "I"),
    8: (torch.uint64, "Q"),
}


def _save_safetensors_pure(tensors: Dict[str, torch.Tensor], path: str) -> None:
    """Minimal safetensors writer used when save_file cannot run (no numpy)."""
    header: Dict[str, Any] = {}
    blobs: List[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        t = tensor.detach().cpu().contiguous()
        if t.dtype not in _ST_DTYPE_NAMES:
            raise ValueError(f"unsupported dtype for safetensors: {t.dtype}")
        view_dtype, typecode = _ST_VIEW[t.element_size()]
        blob = array.array(typecode, t.view(view_dtype).flatten().tolist()).tobytes()
        header[name] = {
            "dtype": _ST_DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)
    header_json = json.dumps(header).encode("utf-8")
    pad = (-len(header_json)) % 8
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for blob in blobs:
            f.write(blob)


def _save_safetensors(tensors: Dict[str, torch.Tensor], path: str) -> None:
    try:
        save_file(tensors, path)
    except ModuleNotFoundError as e:
        if e.name != "numpy":
            raise
        logging.warning("numpy unavailable; using pure-python safetensors writer")
        _save_safetensors_pure(tensors, path)


# --------------------------------------------------------------------------- #
# source inspection
# --------------------------------------------------------------------------- #
def inspect_source(source: str) -> Dict[str, Any]:
    """Print the structure of a kev head.pt or an autojev checkpoint dir."""
    info: Dict[str, Any] = {"source": source}
    if os.path.isdir(source):
        readout = os.path.join(source, "readout.safetensors")
        cfg_path = os.path.join(source, "decision_config.json")
        tensors = load_file(readout)
        info["readout.safetensors"] = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in tensors.items()
        }
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            info["decision_config.json"] = {
                k: (v if not isinstance(v, list) else f"<{len(v)} entries>")
                for k, v in cfg.items()
            }
        return info

    obj = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "head" in obj:
        info["meta"] = {k: v for k, v in obj.items() if k != "head"}
        state = obj["head"]
    else:
        state = obj
    info["tensors"] = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)}
        for k, v in state.items()
    }
    return info


# --------------------------------------------------------------------------- #
# checkpoint patching helpers
# --------------------------------------------------------------------------- #
def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _resolve_token_ids(target_dir: str, token_strings: List[str]) -> Dict[str, int]:
    """Resolve special-token strings to ids using the target ckpt tokenizer files."""
    content_to_id: Dict[str, int] = {}
    tok_json = os.path.join(target_dir, "tokenizer.json")
    if os.path.isfile(tok_json):
        data = _read_json(tok_json)
        for entry in data.get("added_tokens", []):
            content_to_id[entry["content"]] = entry["id"]
    tok_cfg = os.path.join(target_dir, "tokenizer_config.json")
    if os.path.isfile(tok_cfg):
        data = _read_json(tok_cfg)
        for tid, entry in data.get("added_tokens_decoder", {}).items():
            content_to_id.setdefault(entry["content"], int(tid))
    missing = [s for s in token_strings if s not in content_to_id]
    if missing:
        raise ValueError(
            f"special tokens {missing} not found in target checkpoint tokenizer; "
            "the decision markers must be existing tokenizer special tokens"
        )
    return {s: content_to_id[s] for s in token_strings}


def _patch_index(target_dir: str, tensor_files: Dict[str, str], force: bool) -> bool:
    """Add decision_head.* entries to model.safetensors.index.json if one exists.

    RTP-LLM's CkptDatabase enumerates shard files exclusively from the index
    when it exists, so the extra safetensors file must be registered there.
    Returns True when the index was (or had to be) up to date.
    """
    index_path = os.path.join(target_dir, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return False
    index = _read_json(index_path)
    weight_map = index.setdefault("weight_map", {})
    changed = False
    for tensor_name, file_name in tensor_files.items():
        existing = weight_map.get(tensor_name)
        if existing == file_name:
            continue
        if existing is not None and not force:
            raise RuntimeError(
                f"index already maps {tensor_name} -> {existing}; use --force to overwrite"
            )
        weight_map[tensor_name] = file_name
        changed = True
    if not changed:
        logging.info("model.safetensors.index.json already up to date")
        return True
    # keep metadata totals roughly honest (used for progress/sizing only)
    head_path = os.path.join(target_dir, DECISION_HEAD_FILE)
    meta = index.setdefault("metadata", {})
    if os.path.isfile(head_path) and "total_size" in meta:
        meta["total_size"] = int(meta["total_size"]) + os.path.getsize(head_path)
    _write_json(index_path, index)
    return True


def install_decision_head(
    target_dir: str,
    tensors: Dict[str, torch.Tensor],
    decision_block: Dict[str, Any],
    force: bool = False,
) -> None:
    """Write decision_head.safetensors and the config.json "decision" block."""
    if not os.path.isdir(target_dir):
        raise ValueError(f"target checkpoint dir does not exist: {target_dir}")
    config_path = os.path.join(target_dir, "config.json")
    if not os.path.isfile(config_path):
        raise ValueError(f"target checkpoint has no config.json: {target_dir}")

    head_path = os.path.join(target_dir, DECISION_HEAD_FILE)
    new_tensors = {k: v.contiguous() for k, v in tensors.items()}

    if os.path.exists(head_path):
        existing = load_file(head_path)
        same = set(existing) == set(new_tensors) and all(
            torch.equal(existing[k], new_tensors[k]) for k in existing
        )
        if same:
            logging.info("%s already up to date", head_path)
        elif force:
            _save_safetensors(new_tensors, head_path)
            logging.info("overwrote %s (--force)", head_path)
        else:
            raise RuntimeError(
                f"{head_path} exists with different content; use --force to overwrite"
            )
    else:
        _save_safetensors(new_tensors, head_path)
        logging.info("wrote %s", head_path)

    config = _read_json(config_path)
    existing_block = config.get(DECISION_CONFIG_KEY)
    if existing_block is not None:
        if existing_block == decision_block:
            logging.info("config.json %r block already up to date", DECISION_CONFIG_KEY)
        elif force:
            config[DECISION_CONFIG_KEY] = decision_block
            _write_json(config_path, config)
            logging.info("overwrote config.json %r block (--force)", DECISION_CONFIG_KEY)
        else:
            raise RuntimeError(
                "config.json already has a different "
                f"{DECISION_CONFIG_KEY!r} block; use --force to overwrite"
            )
    else:
        config[DECISION_CONFIG_KEY] = decision_block
        _write_json(config_path, config)
        logging.info("added %r block to config.json", DECISION_CONFIG_KEY)

    tensor_files = {name: DECISION_HEAD_FILE for name in new_tensors}
    _patch_index(target_dir, tensor_files, force)


# --------------------------------------------------------------------------- #
# kev (LoRA + pointer head)
# --------------------------------------------------------------------------- #
def convert_kev(
    adapter_dir: str,
    target_dir: str,
    temperature: Optional[float] = None,
    no_temperature: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    head_path = os.path.join(adapter_dir, "head.pt")
    obj = torch.load(head_path, map_location="cpu", weights_only=True)
    if not (isinstance(obj, dict) and isinstance(obj.get("head"), dict)):
        raise ValueError(f"{head_path} is not a kev head checkpoint (no 'head' state dict)")
    state = obj["head"]
    missing = [k for k in POINTER_TENSORS if k not in state]
    if missing:
        raise ValueError(f"{head_path} missing pointer head tensors: {missing}")

    tensors = {POINTER_TENSORS[k][0]: state[k].float() for k in POINTER_TENSORS}
    proj_dim, hidden = tensors["decision_head.q.weight"].shape

    # temperature: explicit flag wins; otherwise reuse the checkpoint's own
    # calibration (eval.json temperature_scaling.T) unless --no-temperature.
    scale_temperature: Optional[float] = temperature
    if scale_temperature is None and not no_temperature:
        eval_path = os.path.join(adapter_dir, "eval.json")
        if os.path.isfile(eval_path):
            fitted = _read_json(eval_path).get("temperature_scaling", {}).get("T")
            if fitted is not None:
                scale_temperature = float(fitted)
                logging.info("using fitted temperature T=%s from eval.json", fitted)
    if scale_temperature is not None and (
        not math.isfinite(scale_temperature) or scale_temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")

    marker_ids = _resolve_token_ids(target_dir, list(KEV_DEFAULT_MARKERS.values()))
    decision_block: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "pointer",
        "prompt_format": "kev_branch",
        "proj_dim": proj_dim,
        "scale": 1.0 / math.sqrt(proj_dim),
        "markers": dict(KEV_DEFAULT_MARKERS),
        "marker_token_ids": {
            name: marker_ids[token] for name, token in KEV_DEFAULT_MARKERS.items()
        },
    }
    for src_key, (_, cfg_key) in POINTER_TENSORS.items():
        decision_block[cfg_key] = POINTER_TENSORS[src_key][0]
    if scale_temperature is not None:
        decision_block["temperature"] = scale_temperature

    install_decision_head(target_dir, tensors, decision_block, force=force)
    return decision_block


# --------------------------------------------------------------------------- #
# autojev (linear readout over answer codes)
# --------------------------------------------------------------------------- #
def convert_autojev(
    ckpt_dir: str,
    target_dir: str,
    temperature: Optional[float] = None,
    force: bool = False,
) -> Dict[str, Any]:
    readout_path = os.path.join(ckpt_dir, "readout.safetensors")
    cfg_path = os.path.join(ckpt_dir, "decision_config.json")
    tensors_src = load_file(readout_path)
    if set(tensors_src) != {"weight"}:
        raise ValueError(f"{readout_path} must contain exactly one 'weight' tensor")
    weight = tensors_src["weight"]
    if weight.ndim != 2:
        raise ValueError("readout weight must have shape [num_labels, hidden]")
    num_labels = weight.shape[0]

    decision_cfg: Dict[str, Any] = {}
    if os.path.isfile(cfg_path):
        decision_cfg = _read_json(cfg_path)
        if decision_cfg.get("format_version") != 1:
            raise ValueError("unsupported decision_config.json format_version")

    scale_temperature = temperature
    if scale_temperature is None:
        fitted = decision_cfg.get("temperature")
        scale_temperature = float(fitted) if fitted is not None else None
    if scale_temperature is not None and (
        not math.isfinite(scale_temperature) or scale_temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")

    codes = decision_cfg.get("codes")
    if codes is not None and len(codes) != num_labels:
        raise ValueError(
            f"decision_config.json codes ({len(codes)}) != readout rows ({num_labels})"
        )

    decision_block: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "linear",
        "prompt_format": "chat",
        "head_weight": LINEAR_WEIGHT_TENSOR[0],
        "num_labels": num_labels,
    }
    if codes is not None:
        decision_block["codes"] = list(codes)
    if scale_temperature is not None:
        decision_block["temperature"] = scale_temperature

    tensors = {LINEAR_WEIGHT_TENSOR[0]: weight}
    install_decision_head(target_dir, tensors, decision_block, force=force)
    return decision_block


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="dump the structure of a source head")
    p_inspect.add_argument("--source", required=True,
                           help="kev head.pt file or autojev checkpoint directory")

    p_kev = sub.add_parser("kev", help="convert a kev LoRA pointer head (head.pt)")
    p_kev.add_argument("--adapter-dir", required=True, help="kev adapter directory")
    p_kev.add_argument("--target-dir", required=True,
                       help="serving checkpoint dir (copy of the Qwen2.5 base)")
    p_kev.add_argument("--temperature", type=float, default=None,
                       help="override calibration temperature")
    p_kev.add_argument("--no-temperature", action="store_true",
                       help="do not bake in eval.json's fitted temperature")
    p_kev.add_argument("--force", action="store_true", help="overwrite existing output")

    p_auto = sub.add_parser("autojev", help="convert an autojev linear readout")
    p_auto.add_argument("--ckpt-dir", required=True,
                        help="autojev checkpoint dir (readout.safetensors + decision_config.json)")
    p_auto.add_argument("--target-dir", required=True,
                        help="checkpoint dir to patch (may equal --ckpt-dir)")
    p_auto.add_argument("--temperature", type=float, default=None,
                        help="override calibration temperature")
    p_auto.add_argument("--force", action="store_true", help="overwrite existing output")

    args = parser.parse_args()
    if args.command == "inspect":
        print(json.dumps(inspect_source(args.source), indent=2, default=str))
    elif args.command == "kev":
        block = convert_kev(args.adapter_dir, args.target_dir,
                            temperature=args.temperature,
                            no_temperature=args.no_temperature, force=args.force)
        print(json.dumps({DECISION_CONFIG_KEY: block}, indent=2))
    elif args.command == "autojev":
        block = convert_autojev(args.ckpt_dir, args.target_dir,
                                temperature=args.temperature, force=args.force)
        print(json.dumps({DECISION_CONFIG_KEY: block}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
