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

# autotrust/JEV-27B-VL: merge the peft LoRA into a copy, then install the
# 264-row verbalizer head (24 trained slots + wide single-token codes)
python convert_decision_head.py jev_vl \
    --ckpt-dir /models/JEV-27B-VL --target-dir /srv/jev-27b-vl-rtp

# Cloudflare/clef: install the 122-tensor joint schema head (text-only by
# default; --keep-vision-config for the multimodal phase)
python convert_decision_head.py clef \
    --ckpt-dir /models/clef --target-dir /srv/clef-rtp
"""

import argparse
import array
import itertools
import json
import logging
import math
import os
import shutil
import string
import struct
import sys
from typing import Any, Dict, List, Optional, Tuple

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
    strip_vision_token_ids(target_dir)
    return decision_block


def strip_vision_token_ids(target_dir: str) -> None:
    """Drop the vision token ids a VL-family config carries.

    The decision path is text-only: with the ids present, the qwen35 config
    parse marks the model multimodal and the embedding engine boots the
    vision tower -- which text checkpoints do not ship, so the first forward
    crashes inside the attention kernels (illegal memory access surfaced in
    the ragged prefill).  Removing the two ids keeps is_multimodal false;
    multimodal decision serving re-adds them when the vision path lands.
    """
    config_path = os.path.join(target_dir, "config.json")
    config = _read_json(config_path)
    stripped = [
        key
        for key in ("vision_start_token_id", "vision_end_token_id")
        if config.pop(key, None) is not None
    ]
    if stripped:
        _write_json(config_path, config)
        logging.info("stripped %s from config.json (text-only decision)", stripped)


# --------------------------------------------------------------------------- #
# jev-27b-vl (LoRA over Qwen3.8-27B + 24-slot verbalizer head)
# --------------------------------------------------------------------------- #
JEV_ADAPTER_SUBDIR = "adapter_vllm"
JEV_MERGE_MARKER = ".jev_lora_merged"
JEV_ADAPTER_PREFIX = "base_model.model."
JEV_DEFAULT_MAX_OPTIONS = 256
JEV_VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"


def _jev_lora_scale(adapter_config: Dict[str, Any]) -> float:
    r = int(adapter_config["r"])
    alpha = float(adapter_config["lora_alpha"])
    if adapter_config.get("use_rslora"):
        return alpha / math.sqrt(r)
    return alpha / r


def _jev_single_token_labels(target_dir: str) -> List[Tuple[str, int]]:
    """Choice label codes with their single-token ids (serve_decide.py rule).

    Prefers the real tokenizer (checks the label also survives the option-line
    context "x\\n{}) y"); falls back to a vocab.json lookup when transformers
    is unavailable or cannot load the checkpoint's tokenizer, which cannot
    check the context.
    """
    candidates = list(string.ascii_uppercase) + [
        a + b for a, b in itertools.product(string.ascii_uppercase, repeat=2)
    ]
    context = "x\n{}) y"
    tok = None
    try:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
        except Exception as e:
            logging.warning(
                "AutoTokenizer could not load %s (%s); resolving jev labels "
                "from vocab.json without the option-line context check",
                target_dir,
                e,
            )
    except ImportError:
        logging.warning(
            "transformers unavailable; resolving jev labels from vocab.json "
            "without the option-line context check"
        )
    if tok is not None:
        out: List[Tuple[str, int]] = []
        for label in candidates:
            ids = tok.encode(label, add_special_tokens=False)
            if len(ids) == 1 and ids[0] in tok.encode(
                context.format(label), add_special_tokens=False
            ):
                out.append((label, ids[0]))
            if len(out) == JEV_DEFAULT_MAX_OPTIONS:
                break
        if len(out) == JEV_DEFAULT_MAX_OPTIONS:
            return out
        # a tokenizer that loads but yields too few single-token labels
        # (e.g. a synthetic or char-level fixture) falls through to vocab.json
        logging.warning(
            "the checkpoint tokenizer only yields %d single-token labels; "
            "falling back to vocab.json",
            len(out),
        )
    vocab = _read_json(os.path.join(target_dir, "vocab.json"))
    out = []
    for label in candidates:
        if label in vocab:
            out.append((label, int(vocab[label])))
        if len(out) == JEV_DEFAULT_MAX_OPTIONS:
            break
    if len(out) < JEV_DEFAULT_MAX_OPTIONS:
        raise ValueError(
            f"vocab.json only yields {len(out)} single-token labels; "
            "install transformers to resolve labels through the tokenizer"
        )
    return out


def _jev_merge_lora(
    target_dir: str, adapter_dir: str, adapter_config: Dict[str, Any], force: bool
) -> None:
    """Fold the peft LoRA into the copied shards in target_dir (in place)."""
    marker = os.path.join(target_dir, JEV_MERGE_MARKER)
    if os.path.exists(marker):
        logging.info("adapter already merged (%s present)", marker)
        return
    if not force and os.path.exists(os.path.join(target_dir, DECISION_HEAD_FILE)):
        raise RuntimeError(
            f"{target_dir} already carries a decision head but no merge marker; "
            "pass --force to merge anyway"
        )

    scale = _jev_lora_scale(adapter_config)
    adapter_tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    index = _read_json(os.path.join(target_dir, "model.safetensors.index.json"))
    weight_map = index["weight_map"]

    # adapter key -> (checkpoint tensor name, A or B)
    pairs: Dict[str, Dict[str, str]] = {}
    for key in adapter_tensors:
        if not key.startswith(JEV_ADAPTER_PREFIX):
            raise ValueError(f"unexpected adapter tensor name: {key}")
        rest = key[len(JEV_ADAPTER_PREFIX):]
        module, ab, suffix = rest.rsplit(".", 2)
        if not ab.startswith("lora_") or suffix != "weight":
            raise ValueError(f"unexpected adapter tensor name: {key}")
        pairs.setdefault(module, {})[ab] = key

    # every checkpoint tensor touched by the adapter, grouped by shard
    shard_deltas: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for module, ab in sorted(pairs.items()):
        if set(ab) != {"lora_A", "lora_B"}:
            raise ValueError(f"adapter module {module} misses its lora pair")
        ckpt_name = module + ".weight"
        shard = weight_map.get(ckpt_name)
        if shard is None:
            raise ValueError(
                f"adapter targets {ckpt_name} which is absent from the checkpoint index"
            )
        shard_deltas.setdefault(shard, {})[ckpt_name] = (
            adapter_tensors[ab["lora_A"]],
            adapter_tensors[ab["lora_B"]],
        )
    consumed = sum(len(v) for v in shard_deltas.values()) * 2
    if consumed != len(adapter_tensors):
        raise RuntimeError(
            f"only {consumed}/{len(adapter_tensors)} adapter tensors were consumed"
        )
    logging.info(
        "merging %d adapter tensor pairs into %d shards (scale=%s)",
        consumed // 2, len(shard_deltas), scale,
    )

    for shard, deltas in sorted(shard_deltas.items()):
        shard_path = os.path.join(target_dir, shard)
        tensors = dict(load_file(shard_path))
        for ckpt_name, (a, b) in deltas.items():
            base = tensors[ckpt_name]
            delta = (b.float() @ a.float()) * scale
            tensors[ckpt_name] = (base.float() + delta).to(base.dtype)
        _save_safetensors(tensors, shard_path)
        logging.info("merged %d tensors into %s", len(deltas), shard)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(f"scale={scale}\n")


def convert_jev_vl(
    ckpt_dir: str,
    target_dir: str,
    force: bool = False,
    no_merge: bool = False,
) -> Dict[str, Any]:
    """Convert autotrust/JEV-27B-VL into a serving checkpoint with a decision block."""
    adapter_dir = os.path.join(ckpt_dir, JEV_ADAPTER_SUBDIR)
    if not os.path.isdir(adapter_dir):
        raise ValueError(f"no {JEV_ADAPTER_SUBDIR}/ in {ckpt_dir}")
    head_cfg = _read_json(os.path.join(adapter_dir, "decision_head.json"))
    adapter_config = _read_json(os.path.join(adapter_dir, "adapter_config.json"))

    calibration_path = os.path.join(ckpt_dir, "calibration.json")
    temperatures: Dict[str, float] = {}
    if os.path.isfile(calibration_path):
        temperatures = {
            k: float(v)
            for k, v in _read_json(calibration_path).get("per_kind", {}).items()
        }

    if os.path.abspath(ckpt_dir) != os.path.abspath(target_dir):
        if os.path.exists(target_dir) and not os.path.isdir(target_dir):
            raise ValueError(f"target path is not a directory: {target_dir}")
        if not os.path.exists(target_dir):
            logging.info("copying %s -> %s", ckpt_dir, target_dir)
            shutil.copytree(ckpt_dir, target_dir)
    else:
        logging.warning("merging in place; keep a backup of %s", ckpt_dir)

    if no_merge:
        logging.info("--no-merge: skipping LoRA merge (target assumed pre-merged)")
    else:
        _jev_merge_lora(target_dir, adapter_dir, adapter_config, force)

    # 24 trained slots (noul/score/choice<=16) + wide single-token codes.
    slots = head_cfg["slots"]
    verbalizers: List[str] = slots["verbalizers"]
    verbalizer_ids: List[int] = head_cfg["verbalizer_ids"]
    choice_labels = _jev_single_token_labels(target_dir)
    if [tid for _, tid in choice_labels[:16]] != verbalizer_ids[8:24]:
        raise ValueError(
            "the checkpoint tokenizer's first 16 choice labels do not match the "
            "head's trained A-P slots"
        )
    wide = choice_labels[16:]
    labels = list(verbalizers) + [lab for lab, _ in wide]
    label_ids = list(verbalizer_ids) + [tid for _, tid in wide]

    index = _read_json(os.path.join(target_dir, "model.safetensors.index.json"))
    lm_head_shard = index["weight_map"]["lm_head.weight"]
    lm_head = load_file(os.path.join(target_dir, lm_head_shard))["lm_head.weight"]
    rows = lm_head.index_select(
        0, torch.tensor(label_ids, dtype=torch.long)
    ).contiguous()

    bias_values = list(head_cfg["bias"]) + [0.0] * (len(labels) - len(head_cfg["bias"]))
    tensors = {
        "decision_head.weight": rows,
        "decision_head.bias": torch.tensor(bias_values, dtype=rows.dtype),
    }

    num_labels = len(labels)
    choice_lo = slots["ranges"]["choice"][0]
    decision_block: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "linear",
        "prompt_format": "bare",
        "head_weight": "decision_head.weight",
        "head_bias": "decision_head.bias",
        "num_labels": num_labels,
        "codes": [lab for lab, _ in choice_labels],
        "labels": labels,
        "label_token_ids": label_ids,
        "slot_ranges": {
            "noul": list(slots["ranges"]["noul"]),
            "score": list(slots["ranges"]["score"]),
            "choice": [choice_lo, num_labels],
        },
        "vision_marker": JEV_VISION_MARKER,
        "max_images_per_prompt": 8,
    }
    if temperatures:
        decision_block["temperatures"] = temperatures

    install_decision_head(target_dir, tensors, decision_block, force=force)
    # The merged LoRA copy keeps the VL family config, whose vision ids would
    # boot the multimodal engine on a text-only decision deployment (see
    # strip_vision_token_ids).
    strip_vision_token_ids(target_dir)
    return decision_block


# --------------------------------------------------------------------------- #
# Cloudflare clef (joint schema head over Qwen3.8-27B)
# --------------------------------------------------------------------------- #
CLEF_HEAD_FILE = "joint_head.safetensors"
CLEF_HEAD_CONFIG_FILE = "joint_head_config.json"
CLEF_SENTINEL_TOKEN = "<|vision_pad|>"
CLEF_HEAD_PREFIX = "decision_head."


def _resolve_digit_token_ids(target_dir: str) -> List[int]:
    """Ids of the single tokens "0".."9" from the checkpoint tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
    ids = []
    for digit in "0123456789":
        encoded = tok.encode(digit, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"digit {digit!r} is not a single token in this tokenizer"
            )
        ids.append(encoded[0])
    return ids


def _expected_joint_keys(head_config: Dict[str, Any]) -> List[str]:
    """State-dict key list of the ClefJointHead layout (import-free)."""
    width = head_config["width"]
    keys: List[str] = []
    for norm in (
        "hidden_norm",
        "field_norm",
        "option_norm",
        "option_summary_norm",
    ):
        keys += [f"{norm}.weight", f"{norm}.bias"]
    for proj in (
        "memory_projection",
        "question_projection",
        "option_question_projection",
        "global_projection",
        "option_context_projection",
        "option_lexical_projection",
    ):
        keys.append(f"{proj}.weight")
    keys.append("type_embedding.weight")
    for layer_index in range(head_config["routing_layers"]):
        prefix = f"evidence_layers.{layer_index}"
        for norm in ("query_norm", "memory_norm", "feedforward_norm"):
            keys += [f"{prefix}.{norm}.weight", f"{prefix}.{norm}.bias"]
        keys += [
            f"{prefix}.attention.in_proj_weight",
            f"{prefix}.attention.in_proj_bias",
            f"{prefix}.attention.out_proj.weight",
            f"{prefix}.attention.out_proj.bias",
        ]
        keys += [
            f"{prefix}.feedforward.0.weight",
            f"{prefix}.feedforward.0.bias",
            f"{prefix}.feedforward.3.weight",
            f"{prefix}.feedforward.3.bias",
        ]
    for layer_index in range(head_config["layers"]):
        prefix = f"layers.{layer_index}"
        for part in ("self_attn", "multihead_attn"):
            keys += [
                f"{prefix}.{part}.in_proj_weight",
                f"{prefix}.{part}.in_proj_bias",
                f"{prefix}.{part}.out_proj.weight",
                f"{prefix}.{part}.out_proj.bias",
            ]
        for norm in ("norm1", "norm2", "norm3"):
            keys += [f"{prefix}.{norm}.weight", f"{prefix}.{norm}.bias"]
        keys += [
            f"{prefix}.linear1.weight",
            f"{prefix}.linear1.bias",
            f"{prefix}.linear2.weight",
            f"{prefix}.linear2.bias",
        ]
    keys += [
        "residual_scorer.0.weight",
        "residual_scorer.0.bias",
        "residual_scorer.3.weight",
        "residual_scorer.3.bias",
        "prior_logit_scale",
        "joint_logit_scale",
        "residual_gate",
    ]
    return keys


def convert_clef(
    ckpt_dir: str,
    target_dir: str,
    max_length: int = 16384,
    response_format: str = "list",
    force: bool = False,
    keep_vision_config: bool = False,
) -> Dict[str, Any]:
    """Convert Cloudflare/clef into a serving checkpoint with a decision block."""
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"clef checkpoint dir does not exist: {ckpt_dir}")
    head_path = os.path.join(ckpt_dir, CLEF_HEAD_FILE)
    head_config_path = os.path.join(ckpt_dir, CLEF_HEAD_CONFIG_FILE)
    if not os.path.isfile(head_path) or not os.path.isfile(head_config_path):
        raise ValueError(
            f"{ckpt_dir} is not a clef checkpoint (needs {CLEF_HEAD_FILE} and "
            f"{CLEF_HEAD_CONFIG_FILE})"
        )
    head_config = _read_json(head_config_path)
    tensors_src = load_file(head_path)
    expected = set(_expected_joint_keys(head_config))
    if set(tensors_src) != expected:
        missing = sorted(expected - set(tensors_src))
        extra = sorted(set(tensors_src) - expected)
        raise ValueError(
            f"joint head tensor set does not match the architecture "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    if os.path.abspath(ckpt_dir) != os.path.abspath(target_dir):
        if os.path.exists(target_dir) and not os.path.isdir(target_dir):
            raise ValueError(f"target path is not a directory: {target_dir}")
        if not os.path.exists(target_dir):
            logging.info("copying %s -> %s", ckpt_dir, target_dir)
            shutil.copytree(ckpt_dir, target_dir)
    else:
        logging.warning("installing in place; keep a backup of %s", ckpt_dir)

    index = _read_json(os.path.join(target_dir, "model.safetensors.index.json"))
    if "lm_head.weight" not in index["weight_map"]:
        raise ValueError("clef needs lm_head.weight in the checkpoint index")

    # The lexical path reads the (untied) lm_head rows over option tokens.
    tensors = {CLEF_HEAD_PREFIX + k: v for k, v in tensors_src.items()}
    sentinel_ids = _resolve_token_ids(target_dir, [CLEF_SENTINEL_TOKEN])
    digit_ids = _resolve_digit_token_ids(target_dir)

    decision_block: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "joint",
        "prompt_format": "clef_schema",
        "head_config": head_config,
        "head_weight_prefix": CLEF_HEAD_PREFIX,
        "lexical_weight": "lm_head.weight",
        "sentinel_token_id": sentinel_ids[CLEF_SENTINEL_TOKEN],
        "table_digit_token_ids": digit_ids,
        "max_length": max_length,
        "response_format": response_format,
        "escape_special_tokens": True,
    }

    install_decision_head(target_dir, tensors, decision_block, force=force)

    # Text-only phase 1: stripping the two vision ids the qwen35 mm config
    # parse reads keeps is_multimodal false, so the engine never loads the
    # vision tower; --keep-vision-config retains it for phase 2.
    if not keep_vision_config:
        config_path = os.path.join(target_dir, "config.json")
        config = _read_json(config_path)
        if "decision" in config:
            stripped = [
                key
                for key in ("vision_start_token_id", "vision_end_token_id")
                if config.pop(key, None) is not None
            ]
            if stripped:
                _write_json(config_path, config)
                logging.info(
                    "stripped %s from config.json (text-only; "
                    "--keep-vision-config to retain)", stripped
                )
    return decision_block


# --------------------------------------------------------------------------- #
# Mapika/decider-35b-a3b (lm_head label rows over packed Answer slots)
# --------------------------------------------------------------------------- #
DECIDER_MAX_OPTIONS = 255
DECIDER_HEAD_PREFIX = "decision_head."


def _decider_label_table(target_dir: str) -> List[Tuple[str, int]]:
    """255 single-token uppercase labels from the checkpoint tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target_dir, trust_remote_code=True)
    import string

    upper = string.ascii_uppercase
    names = list(upper) + [a + b for a in upper for b in upper]
    out: List[Tuple[str, int]] = []
    for name in names:
        ids = tok.encode(name, add_special_tokens=False)
        if len(ids) == 1:
            out.append((name, ids[0]))
        if len(out) == DECIDER_MAX_OPTIONS:
            break
    if len(out) != DECIDER_MAX_OPTIONS:
        raise ValueError(
            f"the checkpoint tokenizer only yields {len(out)} single-token labels"
        )
    return out


def convert_decider(
    ckpt_dir: str,
    target_dir: str,
    force: bool = False,
    keep_vision_config: bool = False,
) -> Dict[str, Any]:
    """Install the decider label head + decision block (in place or a copy)."""
    if not os.path.isdir(ckpt_dir):
        raise ValueError(f"decider checkpoint dir does not exist: {ckpt_dir}")
    config_path = os.path.join(ckpt_dir, "decider_config.json")
    temperature = 1.0
    if os.path.isfile(config_path):
        temperature = float(_read_json(config_path).get("temperature", 1.0))

    if os.path.abspath(ckpt_dir) != os.path.abspath(target_dir):
        if os.path.exists(target_dir) and not os.path.isdir(target_dir):
            raise ValueError(f"target path is not a directory: {target_dir}")
        if not os.path.exists(target_dir):
            logging.info("copying %s -> %s", ckpt_dir, target_dir)
            shutil.copytree(ckpt_dir, target_dir)
    else:
        logging.warning("installing in place; keep a backup of %s", ckpt_dir)

    index = _read_json(os.path.join(target_dir, "model.safetensors.index.json"))
    if "lm_head.weight" not in index["weight_map"]:
        raise ValueError("decider needs lm_head.weight in the checkpoint index")
    labels, label_ids = zip(*_decider_label_table(target_dir))

    lm_head_shard = index["weight_map"]["lm_head.weight"]
    lm_head = load_file(os.path.join(target_dir, lm_head_shard))["lm_head.weight"]
    rows = lm_head.index_select(
        0, torch.tensor(label_ids, dtype=torch.long)
    ).contiguous()

    tensors = {DECIDER_HEAD_PREFIX + "weight": rows}
    sentinel_ids = _resolve_token_ids(target_dir, [CLEF_SENTINEL_TOKEN])
    digit_ids = _resolve_digit_token_ids(target_dir)

    decision_block: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "linear",
        "prompt_format": "decider_schema",
        "head_weight": "decision_head.weight",
        "num_labels": len(labels),
        "codes": list(labels),
        "labels": list(labels),
        "label_token_ids": list(label_ids),
        "temperature": temperature,
        "sentinel_token_id": sentinel_ids[CLEF_SENTINEL_TOKEN],
        "table_digit_token_ids": digit_ids,
        "max_ctx_tokens": 1536,
        "escape_special_tokens": True,
    }

    install_decision_head(target_dir, tensors, decision_block, force=force)

    if not keep_vision_config:
        cfg_path = os.path.join(target_dir, "config.json")
        config = _read_json(cfg_path)
        stripped = [
            key
            for key in ("vision_start_token_id", "vision_end_token_id")
            if config.pop(key, None) is not None
        ]
        if stripped:
            _write_json(cfg_path, config)
            logging.info("stripped %s from config.json", stripped)
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

    p_jev = sub.add_parser(
        "jev_vl", help="convert autotrust/JEV-27B-VL (merge LoRA + verbalizer head)"
    )
    p_jev.add_argument("--ckpt-dir", required=True,
                       help="JEV-27B-VL checkpoint dir (with adapter_vllm/)")
    p_jev.add_argument("--target-dir", required=True,
                       help="serving checkpoint dir (copied from --ckpt-dir if absent)")
    p_jev.add_argument("--no-merge", action="store_true",
                       help="skip the LoRA merge (target already carries merged weights)")
    p_jev.add_argument("--force", action="store_true", help="overwrite existing output")

    p_dec = sub.add_parser(
        "decider", help="convert Mapika/decider-35b-a3b (lm_head label slots)"
    )
    p_dec.add_argument("--ckpt-dir", required=True,
                       help="decider checkpoint dir (decider_config.json + shards)")
    p_dec.add_argument("--target-dir", required=True,
                       help="serving checkpoint dir (copied from --ckpt-dir if absent)")
    p_dec.add_argument("--keep-vision-config", action="store_true",
                       help="keep vision token ids")
    p_dec.add_argument("--force", action="store_true", help="overwrite existing output")

    p_clef = sub.add_parser(
        "clef", help="convert Cloudflare/clef (joint schema head)"
    )
    p_clef.add_argument("--ckpt-dir", required=True,
                        help="clef checkpoint dir (joint_head.safetensors + config)")
    p_clef.add_argument("--target-dir", required=True,
                        help="serving checkpoint dir (copied from --ckpt-dir if absent)")
    p_clef.add_argument("--max-length", type=int, default=16384,
                        help="maximum packed record length (reference default 16384)")
    p_clef.add_argument("--response-format", choices=["list", "systemone"],
                        default="list", help="decision response body shape")
    p_clef.add_argument("--keep-vision-config", action="store_true",
                        help="keep vision token ids (loads the vision tower; phase 2)")
    p_clef.add_argument("--force", action="store_true", help="overwrite existing output")

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
    elif args.command == "jev_vl":
        block = convert_jev_vl(args.ckpt_dir, args.target_dir,
                               force=args.force, no_merge=args.no_merge)
        print(json.dumps({DECISION_CONFIG_KEY: block}, indent=2))
    elif args.command == "decider":
        block = convert_decider(args.ckpt_dir, args.target_dir,
                                force=args.force,
                                keep_vision_config=args.keep_vision_config)
        print(json.dumps({DECISION_CONFIG_KEY: block}, indent=2))
    elif args.command == "clef":
        block = convert_clef(args.ckpt_dir, args.target_dir,
                             max_length=args.max_length,
                             response_format=args.response_format,
                             force=args.force,
                             keep_vision_config=args.keep_vision_config)
        print(json.dumps({DECISION_CONFIG_KEY: block}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
