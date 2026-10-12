"""Native decision-release layouts served without checkpoint conversion.

Some decision models ship in their original release layout rather than the
in-checkpoint ``"decision"`` contract (config block + decision_head tensors
inside the safetensors index).  This module recognizes those layouts on
load and synthesizes the equivalent settings in memory; nothing is written
back to the checkpoint.

Recognized layouts:

* autotrust/JEV-27B-VL -- a self-contained directory with the base shards,
  ``adapter_vllm/`` (peft LoRA + ``decision_head.json`` carrying the
  24-slot verbalizer table: ids, biases, slot ranges) and
  ``calibration.json`` (per-kind temperatures).  The decision logits are
  lm_head rows of the verbalizer token ids plus the bias, so no head
  tensor needs to be materialized: the head reads ``lm_head.weight``
  directly and this source supplies the row selection and bias.
  The LoRA adapter is served through the framework's static LoRA merge
  (``--lora_info '{"jev": "<ckpt>/adapter_vllm"}'``).
* kev-0.5b -- an adapter directory (``head.pt`` pointer head +
  ``adapter_config.json`` peft LoRA + ``eval.json`` fitted temperature)
  over a separate base checkpoint directory.  ``ckpt_path`` points at the
  base and the adapter directory is supplied through the decision source
  resolution (``--decision_source``); its LoRA can likewise be served via
  ``--lora_info``.

Checkpoints that already carry a ``"decision"`` block (autojev-27b,
Cloudflare/clef, decider-35b-a3b) never reach this module.
"""

import json
import logging
import math
import os
import string
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

DECISION_FORMAT_VERSION = 1
JEV_ADAPTER_SUBDIR = "adapter_vllm"
JEV_HEAD_CONFIG = "decision_head.json"
JEV_CALIBRATION = "calibration.json"
JEV_VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
KEV_HEAD_FILE = "head.pt"
KEV_EVAL = "eval.json"
KEV_DEFAULT_MARKERS = {
    "state": "<state>",
    "question": "<question>",
    "option": "<opt>",
    "option_end": "</opt>",
    "decide": "<decide>",
}
# Single-token choice codes beyond the trained A-P slots, in serve_decide.py
# order (upper singles first, then double letters).
_KEV_WIDE_CODES = list(string.ascii_uppercase) + [
    a + b for a, b in product(string.ascii_uppercase, repeat=2)
]


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _positive_temperature(value: Any, source: str) -> Optional[float]:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(t) or t <= 0:
        logging.warning("ignoring non-positive %s temperature %r", source, value)
        return None
    return t


def detect_jev_source(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Synthesize decision settings for an autotrust/JEV-27B-VL release."""
    adapter_dir = os.path.join(ckpt_path, JEV_ADAPTER_SUBDIR)
    head_path = os.path.join(adapter_dir, JEV_HEAD_CONFIG)
    if not os.path.isfile(head_path):
        return None
    head_cfg = _read_json(head_path)
    slots = head_cfg["slots"]
    ranges = slots["ranges"]
    verbalizers: List[str] = slots["verbalizers"]
    verbalizer_ids: List[int] = head_cfg["verbalizer_ids"]
    if len(verbalizers) != len(verbalizer_ids):
        raise ValueError(
            f"{head_path}: verbalizer/slot count mismatch "
            f"({len(verbalizers)} vs {len(verbalizer_ids)})"
        )

    temperatures: Dict[str, float] = {}
    calibration_path = os.path.join(ckpt_path, JEV_CALIBRATION)
    if os.path.isfile(calibration_path):
        per_kind = _read_json(calibration_path).get("per_kind") or {}
        for kind, value in per_kind.items():
            t = _positive_temperature(value, f"jev {kind}")
            if t is not None:
                temperatures[kind] = t

    # The trained choice slots end at the last verbalizer; wider requests
    # fall through to single-token codes resolved against the tokenizer at
    # module construction (see resolve_jev_labels).
    settings: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "linear",
        "prompt_format": "bare",
        "head_weight": "lm_head.weight",
        "head_bias_values": list(head_cfg["bias"]),
        "verbalizers": list(verbalizers),
        "verbalizer_ids": list(verbalizer_ids),
        "slot_ranges": {
            "noul": list(ranges["noul"]),
            "score": list(ranges["score"]),
            "choice": [int(ranges["choice"][0]), None],  # None = as wide as the codes
        },
        "vision_marker": JEV_VISION_MARKER,
        "max_images_per_prompt": 8,
        "_native_source": "jev",
    }
    if temperatures:
        settings["temperatures"] = temperatures
    return settings


def resolve_jev_labels(
    settings: Dict[str, Any], tokenizer: Any
) -> Dict[str, Any]:
    """Fill the wide-code label table using the checkpoint tokenizer.

    The trained slots keep their fitted biases (``head_bias_values`` becomes
    ``head_bias``), the wider single-token codes get a zero bias, and the
    head weight stays ``lm_head.weight`` with ``label_token_ids`` selecting
    rows.  ``codes`` carries the full filtered single-token list the way
    serve_decide.py prints it; its head overlaps the trained slots, so the
    label table appends only the codes beyond them.
    """
    verbalizers: List[str] = settings["verbalizers"]
    verbalizer_ids: List[int] = settings["verbalizer_ids"]
    trained_choice = int(settings["slot_ranges"]["choice"][0])
    wide = _jev_single_token_codes(tokenizer, verbalizer_ids[trained_choice:])
    # serve_decide.py indexes choice options into the filtered list from its
    # head: the first trained-slot-count entries ARE the trained slots
    # (validated inside _jev_single_token_codes), so only the codes beyond
    # them extend the table.  Re-appending the trained letters would place a
    # zero-bias duplicate of row 8+k at row 24+k and shift every wider
    # option onto another letter's lm_head row.
    trained_slots = len(verbalizer_ids) - trained_choice
    beyond = wide[trained_slots:]
    labels = list(verbalizers) + [label for label, _ in beyond]
    label_ids = list(verbalizer_ids) + [tid for _, tid in beyond]
    bias = list(settings.pop("head_bias_values"))
    bias += [0.0] * (len(labels) - len(bias))
    settings = dict(settings)
    settings.pop("verbalizers", None)
    settings.pop("verbalizer_ids", None)
    settings["labels"] = labels
    settings["label_token_ids"] = label_ids
    settings["head_bias"] = bias
    slot_ranges = dict(settings["slot_ranges"])
    slot_ranges["choice"] = [trained_choice, len(labels)]
    settings["slot_ranges"] = slot_ranges
    settings["num_labels"] = len(labels)
    # The renderer prints one code per choice option while the scores read
    # the head rows positionally from slot_ranges.choice, so the code for
    # option j must be the label of row choice[0] + j.  Drawing the codes
    # from the SAME filtered list that feeds label_ids keeps the two in
    # lockstep: a multi-token letter skipped by the tokenizer is absent from
    # both and never desynchronizes the printed letter from the scored
    # lm_head row.
    settings["codes"] = [label for label, _ in wide]
    return settings


def _jev_single_token_codes(
    tokenizer: Any, trained_choice_ids: List[int]
) -> List[Tuple[str, int]]:
    """Wide single-token codes per serve_decide.py: A..Z, AA.., checked in
    the option-line context ``x\\n{}) y`` so the id survives serving."""
    out: List[Tuple[str, int]] = []
    for code in _KEV_WIDE_CODES:
        ids = tokenizer(f"x\n{code}) y", add_special_tokens=False).input_ids
        code_ids = tokenizer(code, add_special_tokens=False).input_ids
        if len(code_ids) == 1 and code_ids[0] in ids:
            out.append((code, int(code_ids[0])))
    if not out:
        raise ValueError("the tokenizer produced no single-token choice codes")
    # The trained slots must agree with the tokenizer's own ids.
    resolved_head = [tid for _, tid in out[: len(trained_choice_ids)]]
    if resolved_head != list(trained_choice_ids):
        raise ValueError(
            "the checkpoint tokenizer's choice codes do not match the "
            "head's trained slots"
        )
    return out


def detect_kev_source(adapter_dir: str) -> Optional[Dict[str, Any]]:
    """Synthesize decision settings for a kev-0.5b adapter directory."""
    head_path = os.path.join(adapter_dir, KEV_HEAD_FILE)
    if not os.path.isfile(head_path):
        return None
    import torch

    obj = torch.load(head_path, map_location="cpu", weights_only=True)
    state = obj.get("head") if isinstance(obj, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"{head_path} is not a kev head checkpoint")
    missing = [k for k in ("q.weight", "q.bias", "k.weight", "k.bias") if k not in state]
    if missing:
        raise ValueError(f"{head_path} missing pointer head tensors: {missing}")
    proj_dim = int(state["q.weight"].shape[0])

    temperature: Optional[float] = None
    eval_path = os.path.join(adapter_dir, KEV_EVAL)
    if os.path.isfile(eval_path):
        fitted = _read_json(eval_path).get("temperature_scaling", {}).get("T")
        temperature = _positive_temperature(fitted, "kev")

    settings: Dict[str, Any] = {
        "format_version": DECISION_FORMAT_VERSION,
        "head_type": "pointer",
        "prompt_format": "kev_branch",
        "proj_dim": proj_dim,
        "scale": 1.0 / math.sqrt(proj_dim),
        "markers": dict(KEV_DEFAULT_MARKERS),
        "head_tensors": {k: v for k, v in state.items()},
        "_native_source": "kev",
    }
    if temperature is not None:
        settings["temperature"] = temperature
    return settings


def resolve_kev_markers(
    settings: Dict[str, Any], tokenizer: Any
) -> Dict[str, Any]:
    marker_ids: Dict[str, int] = {}
    for name, token in KEV_DEFAULT_MARKERS.items():
        ids = tokenizer(token, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"kev marker {token} is not a single token")
        marker_ids[name] = int(ids[0])
    settings = dict(settings)
    settings["marker_token_ids"] = marker_ids
    return settings


def detect_native_decision_source(
    ckpt_path: str, decision_source: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Try the native release layouts: JEV in the checkpoint itself, kev in
    the ``--decision_source`` adapter directory."""
    if decision_source:
        settings = detect_kev_source(decision_source)
        if settings is not None:
            return settings
        raise ValueError(f"no kev head.pt under decision_source {decision_source}")
    return detect_jev_source(ckpt_path)
