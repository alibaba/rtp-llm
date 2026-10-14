"""Validator for EPLB monitoring window artifacts (frozen schema v1).

Scope and limits
----------------
This tool validates that a window artifact is INTERNALLY CONSISTENT and that
its reported per-layer ratios follow the contract formulas, recomputed with
the independent pure-Python Fraction oracle (window_ratio_oracle.py):

    moe_lb        = sum_t( max_r x[t,l,r] ) / sum_t( mean_r x[t,l,r] )
    moe_lb_window = max_r( sum_t x[t,l,r] ) / mean_r( sum_t x[t,l,r] )

It also checks expert->rank conservation under contiguous physical-slot
ownership (rank r owns experts [r*k, (r+1)*k), k = physical_expert_count /
ep_size): per-rank expert activation sums must equal rank_activations, and
each layer's total_activations must equal the sum of its physical expert
histogram.

It does NOT prove that the production router counted real traffic correctly:
the artifact's rank_history is taken as given.  Router truthfulness is
covered by the known-topk synthetic GPU tests and the model smoke harness.

CLI:
    python artifact_validator.py <window.json> [<window2.json> ...]

Validates each artifact and the sequence they form (step-range continuity
across all windows; mapping_epoch never decreases in step order).  Exit code
0 = ok, 1 = validation failure, 2 = usage/IO error.
"""

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from window_ratio_oracle import WindowReport  # noqa: E402

SCHEMA_VERSION = 1
# Reported doubles are compared to exact Fractions with this tolerance.
RATIO_REL_TOL = 1e-9

_INT_METADATA = (
    "model_id",
    "world_rank",
    "world_size",
    "ep_size",
    "layer_count",
    "physical_expert_count",
    "mapping_epoch",
    "step_begin",
    "step_end",
    "sample_count",
    "window_capacity",
)


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_int_matrix(
    value: Any, shape: List[int], name: str, errors: List[str]
) -> Optional[list]:
    """Validate a nested list of non-negative ints with the given shape."""
    if not isinstance(value, list) or len(value) != shape[0]:
        errors.append(f"{name}: expected outer list of length {shape[0]}")
        return None
    if len(shape) == 1:
        for i, v in enumerate(value):
            if not _is_int(v) or v < 0:
                errors.append(f"{name}[{i}]: expected non-negative int")
                return None
        return value
    out = []
    for i, sub in enumerate(value):
        checked = _check_int_matrix(sub, shape[1:], f"{name}[{i}]", errors)
        if checked is None:
            return None
        out.append(checked)
    return out


def _ratio_close(reported: float, expected: Fraction) -> bool:
    if not isinstance(reported, (int, float)) or isinstance(reported, bool):
        return False
    if not math.isfinite(reported):
        return False
    exp = float(expected)
    return abs(reported - exp) <= RATIO_REL_TOL * max(1.0, abs(exp))


def validate_window_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one window artifact; recompute its ratios with the oracle.

    Returns {"ok": bool, "errors": [...], "layers": {layer: {...}}}.
    Never raises on malformed input; all problems land in "errors".
    """
    errors: List[str] = []
    result: Dict[str, Any] = {"ok": False, "errors": errors, "layers": {}}
    if not isinstance(artifact, dict):
        errors.append("artifact is not a JSON object")
        return result

    if (
        not _is_int(artifact.get("schema_version"))
        or artifact["schema_version"] != SCHEMA_VERSION
    ):
        errors.append(
            f"schema_version must be {SCHEMA_VERSION}, "
            f"got {artifact.get('schema_version')!r}"
        )
        return result

    for key in _INT_METADATA:
        if key not in artifact or not _is_int(artifact[key]):
            errors.append(f"metadata {key!r}: missing or not an int")
    if errors:
        return result

    meta = {k: artifact[k] for k in _INT_METADATA}
    ep_size = meta["ep_size"]
    layer_count = meta["layer_count"]
    expert_count = meta["physical_expert_count"]
    sample_count = meta["sample_count"]

    if not (0 <= meta["world_rank"] < max(meta["world_size"], 1)):
        errors.append("world_rank out of range")
    for key in (
        "world_size",
        "ep_size",
        "layer_count",
        "physical_expert_count",
        "window_capacity",
    ):
        if meta[key] <= 0:
            errors.append(f"{key} must be > 0")
    if meta["mapping_epoch"] < 0:
        errors.append("mapping_epoch must be >= 0")
    if meta["step_begin"] < 1:
        errors.append("step_begin must be >= 1 for production forwards")
    if meta["step_end"] < meta["step_begin"]:
        errors.append("step_end < step_begin")
    span = meta["step_end"] - meta["step_begin"] + 1
    if sample_count != span:
        errors.append(
            f"sample_count {sample_count} != step span {span} "
            f"({meta['step_begin']}..{meta['step_end']})"
        )
    if sample_count > meta["window_capacity"]:
        errors.append(
            f"sample_count {sample_count} exceeds window_capacity "
            f"{meta['window_capacity']}"
        )
    if expert_count % max(ep_size, 1) != 0:
        errors.append(
            f"physical_expert_count {expert_count} not divisible by ep_size "
            f"{ep_size}: contiguous expert->rank ownership undefined"
        )
    if errors:
        return result
    experts_per_rank = expert_count // ep_size

    hist = _check_int_matrix(
        artifact.get("physical_expert_activations"),
        [layer_count, expert_count],
        "physical_expert_activations",
        errors,
    )
    history = _check_int_matrix(
        artifact.get("rank_history"),
        [sample_count, layer_count, ep_size],
        "rank_history",
        errors,
    )
    layers = artifact.get("layers")
    if not isinstance(layers, list) or len(layers) != layer_count:
        errors.append(f"layers: expected list of length {layer_count}")
        layers = None
    if errors:
        return result

    # Oracle recomputation over the artifact's own rank_history.
    report = WindowReport(history, layer_count, ep_size)

    seen_layers = set()
    for entry in layers:
        if not isinstance(entry, dict):
            errors.append("layers entry is not an object")
            continue
        layer = entry.get("layer")
        if not _is_int(layer) or not 0 <= layer < layer_count:
            errors.append(f"layers entry has bad layer index {layer!r}")
            continue
        if layer in seen_layers:
            errors.append(f"duplicate layers entry for layer {layer}")
            continue
        seen_layers.add(layer)

        oracle = report.layer_ratios(layer)
        detail: Dict[str, Any] = {
            "oracle_valid": oracle["valid"],
            "oracle_moe_lb": (
                float(oracle["moe_lb"]) if oracle["moe_lb"] is not None else None
            ),
            "oracle_moe_lb_window": (
                float(oracle["moe_lb_window"])
                if oracle["moe_lb_window"] is not None
                else None
            ),
            "oracle_rank_totals": oracle["rank_totals"],
            "oracle_sample_count": oracle["sample_count"],
        }
        result["layers"][layer] = detail

        rank_acts = entry.get("rank_activations")
        if (
            not isinstance(rank_acts, list)
            or len(rank_acts) != ep_size
            or any(not _is_int(v) or v < 0 for v in rank_acts)
        ):
            errors.append(
                f"layer {layer}: rank_activations must be "
                f"{ep_size} non-negative ints"
            )
            continue
        total = entry.get("total_activations")
        if not _is_int(total) or total < 0:
            errors.append(
                f"layer {layer}: total_activations must be a " "non-negative int"
            )
            continue
        valid = entry.get("valid")
        if not isinstance(valid, bool):
            errors.append(f"layer {layer}: valid must be a bool")
            continue

        # rank_activations must equal the window-summed history column.
        if rank_acts != oracle["rank_totals"]:
            errors.append(
                f"layer {layer}: rank_activations {rank_acts} != oracle "
                f"{oracle['rank_totals']}"
            )
        if total != oracle["sample_count"]:
            errors.append(
                f"layer {layer}: total_activations {total} != oracle "
                f"{oracle['sample_count']}"
            )
        # Conservation: physical expert histogram sums to the same totals,
        # under contiguous slot ownership (rank r owns [r*k, (r+1)*k)).
        layer_hist = hist[layer]
        if sum(layer_hist) != total:
            errors.append(
                f"layer {layer}: physical_expert_activations sum "
                f"{sum(layer_hist)} != total_activations {total}"
            )
        for r in range(ep_size):
            seg = sum(layer_hist[r * experts_per_rank : (r + 1) * experts_per_rank])
            if seg != rank_acts[r]:
                errors.append(
                    f"layer {layer}: experts of rank {r} sum to {seg}, "
                    f"rank_activations says {rank_acts[r]}"
                )

        # Ratio semantics: null iff invalid; exact formula when valid.
        if not oracle["valid"]:
            if valid:
                errors.append(
                    f"layer {layer}: reported valid=true but the "
                    "window has zero activations"
                )
            if (
                entry.get("moe_lb") is not None
                or entry.get("moe_lb_window") is not None
            ):
                errors.append(
                    f"layer {layer}: invalid layer must report "
                    "null ratios, not fabricated values"
                )
        else:
            if not valid:
                errors.append(
                    f"layer {layer}: reported valid=false but the "
                    "window has activations"
                )
            if not _ratio_close(entry.get("moe_lb"), oracle["moe_lb"]):
                errors.append(
                    f"layer {layer}: moe_lb {entry.get('moe_lb')!r} != oracle "
                    f"{float(oracle['moe_lb'])}"
                )
            if not _ratio_close(entry.get("moe_lb_window"), oracle["moe_lb_window"]):
                errors.append(
                    f"layer {layer}: moe_lb_window "
                    f"{entry.get('moe_lb_window')!r} != oracle "
                    f"{float(oracle['moe_lb_window'])}"
                )

    if seen_layers != set(range(layer_count)):
        errors.append(
            f"layers entries cover {sorted(seen_layers)}, expected "
            f"0..{layer_count - 1}"
        )

    result["ok"] = not errors
    return result


def validate_window_sequence(artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a sequence of window artifacts from one model instance/rank.

    Checks per-window validity plus: identical model/rank/shape metadata,
    step-range continuity across ALL windows (production step numbering does
    not restart when a remap bumps mapping_epoch; an epoch change just
    flushes a partial window first), mapping_epoch non-decreasing in step
    order, and sample_count == step span <= window_capacity.  Per-epoch
    chains are reported separately so windows of different mapping epochs are
    never interpreted under one mapping.  Does NOT require any minimum
    number of windows -- that is the smoke harness's job.
    """
    errors: List[str] = []
    window_results: List[Dict[str, Any]] = []
    for i, artifact in enumerate(artifacts):
        res = validate_window_artifact(artifact)
        window_results.append(res)
        for e in res["errors"]:
            errors.append(f"window[{i}]: {e}")

    result: Dict[str, Any] = {
        "ok": False,
        "errors": errors,
        "window_count": len(artifacts),
        "windows": window_results,
        "chains": {},
    }
    if not artifacts:
        errors.append("empty artifact sequence")
        return result

    identity_keys = (
        "model_id",
        "world_rank",
        "world_size",
        "ep_size",
        "layer_count",
        "physical_expert_count",
    )
    valid_meta = [
        a
        for a in artifacts
        if isinstance(a, dict)
        and all(
            _is_int(a.get(k))
            for k in identity_keys
            + (
                "mapping_epoch",
                "step_begin",
                "step_end",
                "sample_count",
                "window_capacity",
            )
        )
    ]
    if len(valid_meta) != len(artifacts):
        errors.append(
            "some artifacts lack usable metadata; sequence checks " "skipped for those"
        )
    if valid_meta:
        first = valid_meta[0]
        for key in identity_keys:
            for i, a in enumerate(valid_meta):
                if a[key] != first[key]:
                    errors.append(
                        f"mixed {key} in one sequence: window 0 has "
                        f"{first[key]}, window {i} has {a[key]}"
                    )
                    break

        # A remap ends the prior window, but production step numbering does
        # not restart. Verify continuity across epochs as well as within them.
        ordered = sorted(valid_meta, key=lambda a: (a["step_begin"], a["step_end"]))
        for prev, nxt in zip(ordered, ordered[1:]):
            if nxt["step_begin"] != prev["step_end"] + 1:
                errors.append(
                    f"step ranges are not continuous across windows: "
                    f"{prev['step_end']} -> {nxt['step_begin']}"
                )
            if nxt["mapping_epoch"] < prev["mapping_epoch"]:
                errors.append("mapping_epoch decreases in production step order")

        # Per-epoch summaries retain the mapping interpretation boundary.
        chains: Dict[int, List[Dict[str, Any]]] = {}
        for a in valid_meta:
            chains.setdefault(a["mapping_epoch"], []).append(a)
        for epoch, windows in sorted(chains.items()):
            windows.sort(key=lambda a: (a["step_begin"], a["step_end"]))
            chain_errors = []
            for prev, nxt in zip(windows, windows[1:]):
                if nxt["step_begin"] <= prev["step_end"]:
                    chain_errors.append(
                        f"overlap: [{prev['step_begin']},{prev['step_end']}] "
                        f"then [{nxt['step_begin']},{nxt['step_end']}]"
                    )
                elif nxt["step_begin"] != prev["step_end"] + 1:
                    chain_errors.append(
                        f"gap: [{prev['step_begin']},{prev['step_end']}] then "
                        f"[{nxt['step_begin']},{nxt['step_end']}]"
                    )
            errors.extend(f"epoch {epoch}: {e}" for e in chain_errors)
            result["chains"][epoch] = {
                "windows": len(windows),
                "step_begin": windows[0]["step_begin"],
                "step_end": windows[-1]["step_end"],
                "continuous": not chain_errors,
            }

    result["ok"] = not errors
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate EPLB window artifact JSON files."
    )
    parser.add_argument("artifacts", nargs="+", help="window artifact JSON files")
    args = parser.parse_args(argv)

    artifacts = []
    for path in args.artifacts:
        try:
            with open(path, "r", encoding="utf-8") as f:
                artifacts.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot load {path}: {exc}", file=sys.stderr)
            return 2

    sequence = validate_window_sequence(artifacts)
    print(json.dumps(sequence, indent=2, sort_keys=True))
    return 0 if sequence["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
