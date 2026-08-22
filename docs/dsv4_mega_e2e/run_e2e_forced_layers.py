#!/usr/bin/env python3
"""Capture and compare every DSV4 decoder block on fixed decode prefixes.

This is the layer-trajectory companion to ``run_e2e_forced_logits.py``.
It reuses RTP-LLM's env-gated MOEDBG recorder, restricting it to
``decode_layerXX_out`` tensors.  A single 257-token forced trajectory covers
both 128- and 256-token compression boundaries; only requested checkpoints
are retained.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_e2e_forced_logits as logits_test  # noqa: E402


MODE_ENVS = logits_test.MODE_ENVS
LAYER_NAME_RE = re.compile(r"^decode_layer(\d+)_out$")


@dataclass(frozen=True)
class LayerCase:
    case_id: str
    kind: str
    prompt: str
    input_tokens: int
    output_step: int


def _csv_ints(name: str, default: str) -> list[int]:
    values = [int(value) for value in os.environ.get(name, default).split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return sorted(set(values))


def _out_dir() -> Path:
    return Path(os.environ.get("E2E_OUT", "e2e_forced_layers_out")).resolve()


def _manifest_path(mode: str) -> Path:
    return _out_dir() / f"{mode}.forced_layers.manifest.json"


def _build_suite(ckpt: str) -> dict[str, Any]:
    tokenizer = logits_test._load_tokenizer(ckpt)
    boundary_steps = _csv_ints(
        "E2E_LAYER_BOUNDARY_STEPS", "2,126,127,128,129,254,255,256,257"
    )
    context_lengths = _csv_ints("E2E_LAYER_CONTEXTS", "32,512,2048,8192")
    max_step = max(max(boundary_steps), 2)

    base_ids = tokenizer.encode(logits_test.FORCED_TEXT, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    base_ids = [token_id for token_id in base_ids if token_id != eos_id]
    repeats = math.ceil((max_step + 1) / len(base_ids))
    forced_ids = (base_ids * repeats)[: max_step + 1]
    root_id = tokenizer.bos_token_id
    if root_id is None:
        root_id = forced_ids[0]

    semantic_cases = [
        LayerCase(
            case_id=f"semantic/{case_id}",
            kind="semantic",
            prompt=prompt,
            input_tokens=len(tokenizer.encode(prompt)),
            output_step=2,
        )
        for case_id, prompt in logits_test.SEMANTIC_PROMPTS
    ]
    context_cases = [
        LayerCase(
            case_id=f"context/input_{token_count}",
            kind="context",
            prompt=logits_test._make_exact_length_prompt(tokenizer, token_count),
            input_tokens=token_count,
            output_step=2,
        )
        for token_count in context_lengths
    ]
    boundary_prompt = logits_test._make_exact_length_prompt(tokenizer, 1)
    boundary_cases = [
        LayerCase(
            case_id=f"boundary/output_{output_step}",
            kind="boundary",
            prompt=boundary_prompt,
            input_tokens=1,
            output_step=output_step,
        )
        for output_step in boundary_steps
    ]
    cases = semantic_cases + context_cases + boundary_cases
    public_cases = [
        {
            **asdict(case),
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
        }
        for case in cases
    ]
    for case in public_cases:
        case.pop("prompt")
    fingerprint_payload = {
        "cases": public_cases,
        "forced_ids": forced_ids,
        "vocab_size": len(tokenizer),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "cases": cases,
        "public_cases": public_cases,
        "boundary_cases": boundary_cases,
        "semantic_cases": semantic_cases,
        "context_cases": context_cases,
        "forced_ids": forced_ids,
        "tree": logits_test._build_tree_config(forced_ids, root_id, eos_id),
        "vocab_size": len(tokenizer),
        "fingerprint": fingerprint,
    }


def _configure_server(e2e, ckpt: str, suite: dict[str, Any]) -> None:
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    tree_path = out / "forced_layers_tree.json"
    tree_path.write_text(json.dumps(suite["tree"], separators=(",", ":")))
    relative_tree = os.path.relpath(tree_path, Path(ckpt).resolve())
    max_context = max(case.input_tokens for case in suite["cases"])
    required_seq = max_context + max(case.output_step for case in suite["cases"]) + 64
    max_seq_len = 4096
    while max_seq_len < required_seq:
        max_seq_len *= 2
    e2e.SERVER_ARGS = logits_test._replace_server_args(
        e2e.SERVER_ARGS,
        {
            "--tree_decode_config": relative_tree,
            "--max_seq_len": str(max_seq_len),
            "--concurrency_limit": "2",
            "--max_context_batch_size": "1",
        },
    )


def _dump_step(path: Path) -> int:
    match = re.search(r"_step(\d+)\.pt$", path.name)
    if not match:
        raise RuntimeError(f"unrecognized MOEDBG filename: {path}")
    return int(match.group(1))


def _load_dump(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    layer_names = sorted(
        (name for name in payload.get("tensors", {}) if LAYER_NAME_RE.match(name)),
        key=lambda name: int(LAYER_NAME_RE.match(name).group(1)),
    )
    if not layer_names:
        raise RuntimeError(f"{path} contains no decoder layer tensors")
    if len(layer_names) != len(set(layer_names)):
        raise RuntimeError(f"{path} contains duplicate layer names")
    payload["_layer_names"] = layer_names
    return payload


def _validate_layer_names(
    layer_names: list[str], expected: list[str] | None, context: str
) -> list[str]:
    layer_ids = [int(LAYER_NAME_RE.match(name).group(1)) for name in layer_names]
    if layer_ids != list(range(layer_ids[0], layer_ids[-1] + 1)):
        raise RuntimeError(f"{context}: decoder layer ids are not contiguous: {layer_ids}")
    if expected is not None and layer_names != expected:
        raise RuntimeError(f"{context}: decoder layer names differ from the first capture")
    return layer_names


def _new_dumps(capture_dir: Path, before: set[Path]) -> list[Path]:
    current = set(capture_dir.glob("*.pt"))
    paths = sorted(current - before, key=_dump_step)
    for path in paths:
        _load_dump(path)
    return paths


def _request_case(e2e, case: LayerCase, max_new_tokens: int) -> dict[str, Any]:
    request_case = logits_test.Case(
        case_id=case.case_id,
        kind=case.kind,
        prompt=case.prompt,
        input_tokens=case.input_tokens,
        steps=max_new_tokens,
        full_vocab=False,
    )
    payload = logits_test._post_request(e2e.PORT, request_case, [0])
    output_ids = logits_test._flatten_ints(
        logits_test._find_field(payload, "output_ids")
    )
    return {"payload": payload, "output_ids": output_ids}


def _start_pos(payload: dict[str, Any]) -> list[int]:
    value = payload.get("extra", {}).get("start_pos")
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).to(torch.int64).tolist()
    return np.asarray(value).reshape(-1).astype(np.int64).tolist()


def _retain_dump(
    source: Path,
    selected_dir: Path,
    case: LayerCase,
    records: list[dict[str, Any]],
) -> None:
    safe_name = case.case_id.replace("/", "__")
    destination = selected_dir / f"{safe_name}.pt"
    if destination.exists():
        raise RuntimeError(
            f"refusing to overwrite existing layer capture {destination}; use a new E2E_OUT"
        )
    payload = _load_dump(source)
    source.replace(destination)
    records.append(
        {
            "case_id": case.case_id,
            "kind": case.kind,
            "input_tokens": case.input_tokens,
            "output_step": case.output_step,
            "start_pos": _start_pos(payload),
            "layer_names": payload["_layer_names"],
            "path": str(destination.relative_to(_out_dir())),
        }
    )


def _validate_forced_output(
    output_ids: list[int], forced_ids: list[int], expected_steps: int, case_id: str
) -> None:
    expected = forced_ids[:expected_steps]
    if output_ids != expected:
        raise RuntimeError(
            f"{case_id}: forced output mismatch, got {len(output_ids)} tokens, "
            f"expected {len(expected)}"
        )


def run_mode(mode: str) -> None:
    ckpt = os.environ.get("E2E_CKPT")
    if not ckpt:
        raise SystemExit("E2E_CKPT must point at the DSV4 checkpoint")
    if _manifest_path(mode).exists():
        raise SystemExit(
            f"{_manifest_path(mode)} already exists; use a new E2E_OUT to avoid mixing runs"
        )
    suite = _build_suite(ckpt)
    e2e = logits_test._load_e2e_module()
    _configure_server(e2e, ckpt, suite)

    raw_base = _out_dir() / "layer_raw" / f"{mode}_{os.getpid()}"
    capture_dir = raw_base / mode
    selected_dir = _out_dir() / "layer_selected" / mode
    selected_dir.mkdir(parents=True, exist_ok=True)
    extra_env = {
        **MODE_ENVS[mode],
        "MOEDBG": "1",
        "MOEDBG_DIR": str(raw_base),
        "MOEDBG_CASE": mode,
        "MOEDBG_NAME_REGEX": r"^decode_layer[0-9]+_out$",
        "MOEDBG_FULL_THRESHOLD": "1000000",
        "MOEDBG_MAX_SEQ": "128",
    }
    tag = f"forced_layers_{mode}"
    proc = e2e.start_server(tag, extra_env)
    records: list[dict[str, Any]] = []
    expected_layer_names: list[str] | None = None
    try:
        if not e2e.wait_ready(proc):
            raise RuntimeError(f"{tag} not ready")
        capture_dir.mkdir(parents=True, exist_ok=True)

        ordinary_cases = suite["semantic_cases"] + suite["context_cases"]
        for index, case in enumerate(ordinary_cases, 1):
            print(
                f"[{mode}] layer case {index}/{len(ordinary_cases) + 1}: "
                f"{case.case_id}",
                flush=True,
            )
            before = set(capture_dir.glob("*.pt"))
            result = _request_case(e2e, case, case.output_step)
            _validate_forced_output(
                result["output_ids"], suite["forced_ids"], case.output_step, case.case_id
            )
            dumps = _new_dumps(capture_dir, before)
            if len(dumps) != case.output_step - 1:
                raise RuntimeError(
                    f"{case.case_id}: expected {case.output_step - 1} decode dumps, "
                    f"got {len(dumps)}"
                )
            payload = _load_dump(dumps[-1])
            expected_layer_names = _validate_layer_names(
                payload["_layer_names"], expected_layer_names, case.case_id
            )
            _retain_dump(dumps[-1], selected_dir, case, records)
            for path in dumps[:-1]:
                path.unlink()

        boundary_max = max(case.output_step for case in suite["boundary_cases"])
        boundary_template = suite["boundary_cases"][0]
        print(
            f"[{mode}] layer case {len(ordinary_cases) + 1}/{len(ordinary_cases) + 1}: "
            f"boundary trajectory to output {boundary_max}",
            flush=True,
        )
        before = set(capture_dir.glob("*.pt"))
        result = _request_case(e2e, boundary_template, boundary_max)
        _validate_forced_output(
            result["output_ids"], suite["forced_ids"], boundary_max, "boundary"
        )
        dumps = _new_dumps(capture_dir, before)
        if len(dumps) != boundary_max - 1:
            raise RuntimeError(
                f"boundary: expected {boundary_max - 1} decode dumps, got {len(dumps)}"
            )
        retained: set[Path] = set()
        for case in suite["boundary_cases"]:
            source = dumps[case.output_step - 2]
            payload = _load_dump(source)
            expected_layer_names = _validate_layer_names(
                payload["_layer_names"], expected_layer_names, case.case_id
            )
            _retain_dump(source, selected_dir, case, records)
            retained.add(source)
        for path in dumps:
            if path not in retained and path.exists():
                path.unlink()
    finally:
        e2e.stop_server(proc)

    shutil.rmtree(raw_base, ignore_errors=True)
    records.sort(key=lambda record: record["case_id"])
    manifest = {
        "format_version": 1,
        "mode": mode,
        "env": MODE_ENVS[mode],
        "suite_fingerprint": suite["fingerprint"],
        "vocab_size": suite["vocab_size"],
        "forced_ids": suite["forced_ids"],
        "cases": suite["public_cases"],
        "records": records,
    }
    _manifest_path(mode).write_text(json.dumps(manifest, indent=2))
    print(f"[{mode}] saved {_manifest_path(mode)} records={len(records)}")


def _tensor_metrics(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if baseline.shape != candidate.shape:
        raise RuntimeError(f"layer tensor shape mismatch {baseline.shape} != {candidate.shape}")
    left = baseline.float().reshape(-1).numpy().astype(np.float64, copy=False)
    right = candidate.float().reshape(-1).numpy().astype(np.float64, copy=False)
    delta = left - right
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    delta_norm = float(np.linalg.norm(delta))
    denom = left_norm * left_norm + right_norm * right_norm
    calc_diff = 0.0 if denom == 0 else 1.0 - 2.0 * float(np.dot(left, right)) / denom
    cosine_denom = left_norm * right_norm
    cosine = 1.0 if cosine_denom == 0 and delta_norm == 0 else (
        0.0 if cosine_denom == 0 else float(np.dot(left, right)) / cosine_denom
    )
    return {
        "calc_diff": max(calc_diff, 0.0),
        "relative_l2": 0.0 if left_norm == 0 else delta_norm / left_norm,
        "cosine": cosine,
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "baseline_norm": left_norm,
        "candidate_norm": right_norm,
    }


def _load_manifest(mode: str) -> dict[str, Any]:
    path = _manifest_path(mode)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _load_selected(record: dict[str, Any]) -> dict[str, Any]:
    return _load_dump(_out_dir() / record["path"])


def _available_candidates() -> list[str]:
    return [
        mode
        for mode in ("baseline2", "hca", "csa", "mega")
        if _manifest_path(mode).exists()
    ]


def compare_layers(candidates: list[str]) -> None:
    baseline = _load_manifest("baseline")
    if not candidates:
        candidates = _available_candidates()
    if not candidates:
        raise SystemExit("no candidate layer manifests found")
    baseline_records = {record["case_id"]: record for record in baseline["records"]}
    rows: list[dict[str, Any]] = []

    for candidate_name in candidates:
        candidate = _load_manifest(candidate_name)
        if candidate["suite_fingerprint"] != baseline["suite_fingerprint"]:
            raise RuntimeError(f"{candidate_name}: suite fingerprint differs from baseline")
        candidate_records = {
            record["case_id"]: record for record in candidate["records"]
        }
        if candidate_records.keys() != baseline_records.keys():
            raise RuntimeError(f"{candidate_name}: layer case set differs from baseline")
        for case_id, base_record in baseline_records.items():
            cand_record = candidate_records[case_id]
            if base_record["start_pos"] != cand_record["start_pos"]:
                raise RuntimeError(f"{candidate_name}/{case_id}: start_pos differs")
            base_dump = _load_selected(base_record)
            cand_dump = _load_selected(cand_record)
            if base_dump["_layer_names"] != cand_dump["_layer_names"]:
                raise RuntimeError(f"{candidate_name}/{case_id}: layer names differ")
            for name in base_dump["_layer_names"]:
                layer = int(LAYER_NAME_RE.match(name).group(1))
                metrics = _tensor_metrics(
                    base_dump["tensors"][name], cand_dump["tensors"][name]
                )
                rows.append(
                    {
                        "candidate": candidate_name,
                        "case_id": case_id,
                        "kind": base_record["kind"],
                        "input_tokens": base_record["input_tokens"],
                        "output_step": base_record["output_step"],
                        "start_pos": ";".join(map(str, base_record["start_pos"])),
                        "layer": layer,
                        **metrics,
                    }
                )

    fieldnames = list(rows[0].keys())
    csv_path = _out_dir() / "forced_layers_full_comparison.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    case_summary = []
    for candidate_name in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate_name]
        for case_id in sorted({row["case_id"] for row in candidate_rows}):
            case_rows = [row for row in candidate_rows if row["case_id"] == case_id]
            worst = max(case_rows, key=lambda row: row["calc_diff"])
            final = max(case_rows, key=lambda row: row["layer"])
            first_changed = next(
                (row["layer"] for row in case_rows if row["max_abs"] > 0), None
            )
            case_summary.append(
                {
                    "candidate": candidate_name,
                    "case_id": case_id,
                    "input_tokens": case_rows[0]["input_tokens"],
                    "output_step": case_rows[0]["output_step"],
                    "first_changed_layer": first_changed,
                    "worst_layer": worst["layer"],
                    "worst_calc_diff": worst["calc_diff"],
                    "worst_max_abs": max(row["max_abs"] for row in case_rows),
                    "final_layer_calc_diff": final["calc_diff"],
                    "final_layer_max_abs": final["max_abs"],
                }
            )

    layer_summary = []
    for candidate_name in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate_name]
        for layer in sorted({row["layer"] for row in candidate_rows}):
            layer_rows = [row for row in candidate_rows if row["layer"] == layer]
            worst = max(layer_rows, key=lambda row: row["calc_diff"])
            layer_summary.append(
                {
                    "candidate": candidate_name,
                    "layer": layer,
                    "max_calc_diff": worst["calc_diff"],
                    "mean_calc_diff": float(
                        np.mean([row["calc_diff"] for row in layer_rows])
                    ),
                    "max_abs": max(row["max_abs"] for row in layer_rows),
                    "worst_case": worst["case_id"],
                }
            )

    overall_summary = []
    noisy_case_ids = {
        row["case_id"]
        for row in case_summary
        if row["candidate"] == "baseline2" and row["first_changed_layer"] is not None
    }
    for candidate_name in candidates:
        candidate_rows = [row for row in rows if row["candidate"] == candidate_name]
        candidate_cases = [
            row for row in case_summary if row["candidate"] == candidate_name
        ]
        worst = max(candidate_rows, key=lambda row: row["calc_diff"])
        worst_final = max(
            candidate_cases, key=lambda row: row["final_layer_calc_diff"]
        )
        changed_layers = [
            row["first_changed_layer"]
            for row in candidate_cases
            if row["first_changed_layer"] is not None
            and row["case_id"] not in noisy_case_ids
        ]
        overall_summary.append(
            {
                "candidate": candidate_name,
                "cases": len(candidate_cases),
                "layer_rows": len(candidate_rows),
                "first_changed_zero_noise_case": min(changed_layers, default=None),
                "max_calc_diff": worst["calc_diff"],
                "max_calc_diff_case": worst["case_id"],
                "max_calc_diff_layer": worst["layer"],
                "max_final_layer_calc_diff": worst_final["final_layer_calc_diff"],
                "max_final_layer_case": worst_final["case_id"],
            }
        )

    summary = {
        "overall_summary": overall_summary,
        "case_summary": case_summary,
        "layer_summary": layer_summary,
    }
    summary_path = _out_dir() / "forced_layers_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    markdown_path = _out_dir() / "forced_layers_summary.md"
    with markdown_path.open("w") as handle:
        handle.write("# DSV4 forced-prefix layer comparison\n\n")
        handle.write("## Overall\n\n")
        handle.write(
            "| candidate | cases | layer rows | first changed (zero-noise cases) | "
            "max calc_diff | "
            "case | layer | max final calc_diff | case |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|---:|---:|---|\n")
        for row in overall_summary:
            handle.write(
                f"| {row['candidate']} | {row['cases']} | {row['layer_rows']} | "
                f"{row['first_changed_zero_noise_case']} | "
                f"{row['max_calc_diff']:.6e} | "
                f"{row['max_calc_diff_case']} | {row['max_calc_diff_layer']} | "
                f"{row['max_final_layer_calc_diff']:.6e} | "
                f"{row['max_final_layer_case']} |\n"
            )
        handle.write("\n")
        handle.write("## Per case\n\n")
        handle.write(
            "| candidate | case | input | output step | first changed | worst layer | "
            "worst calc_diff | worst max_abs | final calc_diff | final max_abs |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in case_summary:
            handle.write(
                f"| {row['candidate']} | {row['case_id']} | {row['input_tokens']} | "
                f"{row['output_step']} | {row['first_changed_layer']} | "
                f"{row['worst_layer']} | {row['worst_calc_diff']:.6e} | "
                f"{row['worst_max_abs']:.6f} | {row['final_layer_calc_diff']:.6e} | "
                f"{row['final_layer_max_abs']:.6f} |\n"
            )
        handle.write("\n## Per layer\n\n")
        handle.write(
            "| candidate | layer | max calc_diff | mean calc_diff | max_abs | worst case |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---|\n")
        for row in layer_summary:
            handle.write(
                f"| {row['candidate']} | {row['layer']} | {row['max_calc_diff']:.6e} | "
                f"{row['mean_calc_diff']:.6e} | {row['max_abs']:.6f} | "
                f"{row['worst_case']} |\n"
            )
    print(f"full table: {csv_path}")
    print(f"summary: {markdown_path}")
    print(f"summary json: {summary_path}")


def self_test() -> None:
    metrics = _tensor_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert metrics["calc_diff"] < 1e-15
    assert metrics["relative_l2"] == 0.0
    assert abs(metrics["cosine"] - 1.0) < 1e-15
    print("self-test PASS")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_e2e_forced_layers.py "
            "baseline|baseline2|hca|csa|mega|compare|selftest [candidates]"
        )
    action = sys.argv[1]
    if action in MODE_ENVS:
        if len(sys.argv) != 2:
            raise SystemExit(f"{action} takes no positional arguments")
        run_mode(action)
        return
    if action == "compare":
        compare_layers(sys.argv[2:])
        return
    if action == "selftest":
        self_test()
        return
    raise SystemExit(f"unknown action: {action}")


if __name__ == "__main__":
    main()
