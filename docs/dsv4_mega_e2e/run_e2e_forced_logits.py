#!/usr/bin/env python3
"""Compare DSV4 baseline and mega logits on identical decode prefixes.

The server's tree decoder forces one deterministic token at every decode
step.  ``return_logits`` still returns the raw model logits before the tree
mask is applied, so baseline and candidate logits remain comparable even when
their natural greedy tokens would have diverged.

Run modes create one result artifact each::

    python run_e2e_forced_logits.py baseline
    python run_e2e_forced_logits.py baseline2
    python run_e2e_forced_logits.py hca
    python run_e2e_forced_logits.py csa
    python run_e2e_forced_logits.py mega
    python run_e2e_forced_logits.py compare

Required environment for run modes: ``E2E_CKPT``.  ``E2E_GPU``,
``E2E_PYTHON``, ``E2E_OUT`` and ``E2E_JIT_CACHE`` have the same meaning as in
``run_e2e_compare.py``.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import math
import os
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODE_ENVS = {
    "baseline": {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "0"},
    "baseline2": {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "0"},
    "hca": {"DSV4_MEGA_CSA": "0", "DSV4_MEGA_HCA": "1"},
    "csa": {"DSV4_MEGA_CSA": "1", "DSV4_MEGA_HCA": "0"},
    "mega": {"DSV4_MEGA_CSA": "1", "DSV4_MEGA_HCA": "1"},
}
VOCAB_PROBE_SIZE = int(os.environ.get("E2E_LOGITS_PROBE_SIZE", "1024"))
SEMANTIC_STEPS = int(os.environ.get("E2E_LOGITS_SEMANTIC_STEPS", "8"))
BATCH_STEPS = int(os.environ.get("E2E_LOGITS_BATCH_STEPS", "8"))
FORCED_TEXT = (
    "Numerical validation keeps both decode paths on exactly the same token "
    "prefix. The next token is deliberately fixed for reproducible testing. "
)
SEMANTIC_PROMPTS = (
    ("factual", "What is the capital of France?"),
    ("arithmetic", "A shop has 37 boxes with 24 items each. Compute the total."),
    (
        "technical",
        "Explain how paged attention manages a KV cache during batched decode.",
    ),
    (
        "multilingual",
        "Translate this sentence to Chinese: Numerical kernels require careful validation.",
    ),
)


@dataclass(frozen=True)
class Case:
    case_id: str
    kind: str
    prompt: str
    input_tokens: int
    steps: int
    batch: int = 1
    full_vocab: bool = True


def _csv_ints(name: str, default: str) -> list[int]:
    values = [int(value) for value in os.environ.get(name, default).split(",")]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return sorted(set(values))


def _out_dir() -> Path:
    return Path(os.environ.get("E2E_OUT", "e2e_forced_logits_out")).resolve()


def _artifact_paths(label: str) -> tuple[Path, Path]:
    out = _out_dir()
    return (
        out / f"{label}.forced_logits.meta.json",
        out / f"{label}.forced_logits.npz",
    )


def _load_e2e_module():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    return importlib.import_module("run_e2e_compare")


def _load_tokenizer(ckpt: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)


def _make_exact_length_prompt(tokenizer, token_count: int) -> str:
    # DeepSeek's tokenizer round-trips repeated " x" as exactly one token per
    # repetition.  Verify that property instead of silently assuming it.
    unit_ids = tokenizer.encode(" x", add_special_tokens=False)
    if len(unit_ids) != 1:
        raise RuntimeError(f"expected ' x' to encode as one token, got {unit_ids}")
    prompt = tokenizer.decode(unit_ids * token_count, skip_special_tokens=False)
    actual = tokenizer.encode(prompt)
    if len(actual) != token_count:
        raise RuntimeError(
            f"failed to construct exact {token_count}-token prompt: got {len(actual)}"
        )
    return prompt


def _build_tree_config(
    forced_ids: list[int], root_token_id: int, end_token_id: int
) -> dict[str, Any]:
    state = str(root_token_id)
    prefix_dict: dict[str, list[int]] = {}
    for token_id in forced_ids:
        prefix_dict[state] = [token_id]
        state = f"{state}_{token_id}"
    return {
        "start_token_id": root_token_id,
        "end_token_id": end_token_id,
        "sep": "_",
        "prefix_dict": prefix_dict,
    }


def _build_suite(ckpt: str) -> dict[str, Any]:
    tokenizer = _load_tokenizer(ckpt)
    boundary_steps = _csv_ints(
        "E2E_LOGITS_BOUNDARY_STEPS", "1,126,127,128,129,254,255,256,257"
    )
    context_lengths = _csv_ints("E2E_LOGITS_CONTEXTS", "512,2048,8192")
    batches = _csv_ints("E2E_LOGITS_BATCHES", "1,8,32,64,96")
    max_steps = max(max(boundary_steps), SEMANTIC_STEPS, BATCH_STEPS)

    base_forced_ids = tokenizer.encode(FORCED_TEXT, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    base_forced_ids = [token for token in base_forced_ids if token != eos_id]
    if not base_forced_ids:
        raise RuntimeError("forced text produced no usable tokens")
    repeats = math.ceil((max_steps + 1) / len(base_forced_ids))
    forced_ids = (base_forced_ids * repeats)[: max_steps + 1]
    root_id = tokenizer.bos_token_id
    if root_id is None:
        root_id = forced_ids[0]
    if eos_id is None:
        eos_id = 1

    cases: list[Case] = []
    for case_id, prompt in SEMANTIC_PROMPTS:
        cases.append(
            Case(
                case_id=f"semantic/{case_id}",
                kind="semantic",
                prompt=prompt,
                input_tokens=len(tokenizer.encode(prompt)),
                steps=SEMANTIC_STEPS,
            )
        )

    boundary_prompt = _make_exact_length_prompt(tokenizer, 1)
    for steps in boundary_steps:
        cases.append(
            Case(
                case_id=f"boundary/output_{steps}",
                kind="boundary",
                prompt=boundary_prompt,
                input_tokens=1,
                steps=steps,
            )
        )

    for token_count in context_lengths:
        cases.append(
            Case(
                case_id=f"context/input_{token_count}",
                kind="context",
                prompt=_make_exact_length_prompt(tokenizer, token_count),
                input_tokens=token_count,
                steps=2,
            )
        )

    batch_prompt = SEMANTIC_PROMPTS[2][1]
    batch_prompt_tokens = len(tokenizer.encode(batch_prompt))
    for batch in batches:
        cases.append(
            Case(
                case_id=f"batch/b{batch}",
                kind="batch",
                prompt=batch_prompt,
                input_tokens=batch_prompt_tokens,
                steps=BATCH_STEPS,
                batch=batch,
                full_vocab=batch == 1,
            )
        )

    vocab_size = len(tokenizer)
    probe_ids = np.linspace(
        0, vocab_size - 1, min(VOCAB_PROBE_SIZE, vocab_size), dtype=np.int64
    ).tolist()
    probe_ids = sorted(set(probe_ids + forced_ids + [root_id, eos_id]))
    tree = _build_tree_config(forced_ids, root_id, eos_id)
    public_cases = [
        {
            "case_id": case.case_id,
            "kind": case.kind,
            "input_tokens": case.input_tokens,
            "steps": case.steps,
            "batch": case.batch,
            "full_vocab": case.full_vocab,
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
        }
        for case in cases
    ]
    fingerprint_payload = {
        "cases": public_cases,
        "forced_ids": forced_ids,
        "probe_ids": probe_ids,
        "vocab_size": vocab_size,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "tokenizer": tokenizer,
        "cases": cases,
        "public_cases": public_cases,
        "forced_ids": forced_ids,
        "probe_ids": probe_ids,
        "vocab_size": vocab_size,
        "tree": tree,
        "fingerprint": fingerprint,
    }


def _replace_server_args(args: list[str], replacements: dict[str, str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in replacements:
            index += 2
            continue
        result.append(arg)
        index += 1
    for key, value in replacements.items():
        result.extend([key, value])
    return result


def _configure_server(e2e, ckpt: str, suite: dict[str, Any]) -> None:
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)
    tree_path = out / "forced_decode_tree.json"
    tree_path.write_text(json.dumps(suite["tree"], separators=(",", ":")))
    relative_tree_path = os.path.relpath(tree_path, Path(ckpt).resolve())

    max_input = max(case.input_tokens for case in suite["cases"])
    max_steps = max(case.steps for case in suite["cases"])
    required_seq = max_input + max_steps + 64
    max_seq_len = 4096
    while max_seq_len < required_seq:
        max_seq_len *= 2
    max_batch = max(case.batch for case in suite["cases"])
    e2e.SERVER_ARGS = _replace_server_args(
        e2e.SERVER_ARGS,
        {
            "--tree_decode_config": relative_tree_path,
            "--max_seq_len": str(max_seq_len),
            "--concurrency_limit": str(max_batch * 2),
            "--max_context_batch_size": str(min(max_batch, 8)),
        },
    )


def _find_field(payload: dict[str, Any], name: str):
    value = payload.get(name)
    if value is not None:
        return value
    aux = payload.get("aux_info")
    if isinstance(aux, dict):
        return aux.get(name)
    return None


def _post_request(
    port: int,
    case: Case,
    selected_ids: list[int] | None,
) -> dict[str, Any]:
    generate_config: dict[str, Any] = {
        "max_new_tokens": case.steps,
        "top_k": 1,
        "top_p": 0,
        "ignore_eos": True,
        "return_logits": True,
        "return_input_ids": True,
        "return_output_ids": True,
        "aux_info": True,
    }
    if selected_ids is not None:
        generate_config["select_tokens_id"] = selected_ids
    body = json.dumps(
        {"prompt": case.prompt, "generate_config": generate_config}
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def _flatten_ints(value: Any) -> list[int]:
    if value is None:
        return []
    return np.asarray(value, dtype=np.int64).reshape(-1).tolist()


def _capture_payload(
    payload: dict[str, Any],
    case: Case,
    replica: int,
    forced_ids: list[int],
    selected_ids: list[int] | None,
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> None:
    logits = _find_field(payload, "logits")
    if logits is None:
        raise RuntimeError(f"{case.case_id} replica {replica}: response has no logits")
    array = np.asarray(logits, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError(f"{case.case_id} replica {replica}: invalid logits")

    output_ids = _flatten_ints(_find_field(payload, "output_ids"))
    expected = forced_ids[: case.steps]
    if output_ids != expected:
        mismatch = next(
            (
                index
                for index, (actual, wanted) in enumerate(zip(output_ids, expected))
                if actual != wanted
            ),
            min(len(output_ids), len(expected)),
        )
        raise RuntimeError(
            f"{case.case_id} replica {replica}: forced output mismatch at {mismatch}; "
            f"got_len={len(output_ids)} expected_len={len(expected)}"
        )

    input_ids = _flatten_ints(_find_field(payload, "input_ids"))
    if input_ids and len(input_ids) != case.input_tokens:
        raise RuntimeError(
            f"{case.case_id}: server input length {len(input_ids)} != "
            f"planned {case.input_tokens}"
        )

    array_key = f"a{len(arrays):05d}"
    arrays[array_key] = array
    records.append(
        {
            "record_id": f"{case.case_id}/r{replica}",
            "case_id": case.case_id,
            "kind": case.kind,
            "replica": replica,
            "batch": case.batch,
            "input_tokens": case.input_tokens,
            "steps": case.steps,
            "full_vocab": selected_ids is None,
            "selected_ids": selected_ids,
            "logits_shape": list(array.shape),
            "array_key": array_key,
            "response_preview": str(payload.get("response", ""))[:80],
        }
    )


def _run_case(
    e2e,
    case: Case,
    suite: dict[str, Any],
    arrays: dict[str, np.ndarray],
    records: list[dict[str, Any]],
) -> None:
    selected_ids = None if case.full_vocab else suite["probe_ids"]
    if case.batch == 1:
        payload = _post_request(e2e.PORT, case, selected_ids)
        _capture_payload(
            payload,
            case,
            0,
            suite["forced_ids"],
            selected_ids,
            arrays,
            records,
        )
        return

    start = threading.Event()

    def request_one(replica: int):
        start.wait()
        return replica, _post_request(e2e.PORT, case, selected_ids)

    with concurrent.futures.ThreadPoolExecutor(max_workers=case.batch) as executor:
        futures = [executor.submit(request_one, replica) for replica in range(case.batch)]
        start.set()
        payloads = [future.result() for future in futures]
    for replica, payload in sorted(payloads):
        _capture_payload(
            payload,
            case,
            replica,
            suite["forced_ids"],
            selected_ids,
            arrays,
            records,
        )


def run_mode(mode: str) -> None:
    ckpt = os.environ.get("E2E_CKPT")
    if not ckpt:
        raise SystemExit("E2E_CKPT must point at the DSV4 checkpoint")
    suite = _build_suite(ckpt)
    e2e = _load_e2e_module()
    _configure_server(e2e, ckpt, suite)
    tag = f"forced_logits_{mode}"
    proc = e2e.start_server(tag, MODE_ENVS[mode])
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    try:
        if not e2e.wait_ready(proc):
            raise RuntimeError(f"{tag} not ready")
        for index, case in enumerate(suite["cases"], 1):
            print(
                f"[{mode}] {index}/{len(suite['cases'])} {case.case_id} "
                f"B={case.batch} input={case.input_tokens} steps={case.steps}",
                flush=True,
            )
            _run_case(e2e, case, suite, arrays, records)
    finally:
        e2e.stop_server(proc)

    meta_path, array_path = _artifact_paths(mode)
    np.savez_compressed(array_path, **arrays)
    metadata = {
        "format_version": 1,
        "mode": mode,
        "env": MODE_ENVS[mode],
        "suite_fingerprint": suite["fingerprint"],
        "vocab_size": suite["vocab_size"],
        "forced_ids": suite["forced_ids"],
        "probe_ids": suite["probe_ids"],
        "cases": suite["public_cases"],
        "records": records,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2))
    print(f"[{mode}] saved {meta_path} and {array_path}")


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(values.max())
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def _row_metrics(left: np.ndarray, right: np.ndarray, full_vocab: bool) -> dict[str, Any]:
    x = left.astype(np.float64, copy=False)
    y = right.astype(np.float64, copy=False)
    delta = x - y
    denom = float(np.dot(x, x) + np.dot(y, y))
    calc_diff = 0.0 if denom == 0 else 1.0 - 2.0 * float(np.dot(x, y)) / denom
    result: dict[str, Any] = {
        "calc_diff": max(calc_diff, 0.0),
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }
    if not full_vocab:
        return result

    left_top = int(x.argmax())
    right_top = int(y.argmax())
    top5_size = min(5, x.size)
    top10_size = min(10, x.size)
    sorted_left = np.argpartition(x, -top10_size)[-top10_size:]
    sorted_right = np.argpartition(y, -top10_size)[-top10_size:]
    left_top5 = set(np.argpartition(x, -top5_size)[-top5_size:].tolist())
    right_top5 = set(np.argpartition(y, -top5_size)[-top5_size:].tolist())
    top10_overlap = len(set(sorted_left.tolist()) & set(sorted_right.tolist()))
    top_values = np.partition(x, -2)[-2:]
    margin = float(top_values.max() - top_values.min())

    log_p = x - _logsumexp(x)
    log_q = y - _logsumexp(y)
    p = np.exp(log_p)
    q = np.exp(log_q)
    log_mixture = np.logaddexp(log_p, log_q) - math.log(2.0)
    js = 0.5 * float(np.sum(p * (log_p - log_mixture))) + 0.5 * float(
        np.sum(q * (log_q - log_mixture))
    )
    top1_same = left_top == right_top
    result.update(
        {
            "js_divergence": max(js, 0.0),
            "top1_same": top1_same,
            "baseline_top1": left_top,
            "candidate_top1": right_top,
            "baseline_margin": margin,
            "top5_overlap": len(left_top5 & right_top5),
            "top10_overlap": top10_overlap,
            "top1_flip": not top1_same,
            "flip_explained_by_max_abs": top1_same
            or margin <= 2.0 * result["max_abs"],
        }
    )
    return result


def _record_metrics(
    left: np.ndarray, right: np.ndarray, record: dict[str, Any]
) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {left.shape} != {right.shape}")
    width = len(record["selected_ids"] or [])
    if record["full_vocab"]:
        width = left.shape[-1]
    if width <= 0 or left.size % width:
        raise ValueError(f"cannot interpret logits shape {left.shape} with width {width}")
    left_rows = left.reshape(-1, width)
    right_rows = right.reshape(-1, width)
    rows = [
        _row_metrics(x, y, record["full_vocab"])
        for x, y in zip(left_rows, right_rows)
    ]
    result = {
        "record_id": record["record_id"],
        "case_id": record["case_id"],
        "kind": record["kind"],
        "batch": record["batch"],
        "full_vocab": record["full_vocab"],
        "calc_diff": max(row["calc_diff"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "mean_abs": max(row["mean_abs"] for row in rows),
        "rmse": max(row["rmse"] for row in rows),
    }
    if record["full_vocab"]:
        result.update(
            {
                "js_divergence": max(row["js_divergence"] for row in rows),
                "top1_same": all(row["top1_same"] for row in rows),
                "baseline_margin": min(row["baseline_margin"] for row in rows),
                "top5_overlap": min(row["top5_overlap"] for row in rows),
                "top10_overlap": min(row["top10_overlap"] for row in rows),
                "top1_flip": any(row["top1_flip"] for row in rows),
                "flip_explained_by_max_abs": all(
                    row["flip_explained_by_max_abs"] for row in rows
                ),
            }
        )
    return result


def _load_artifact(label: str):
    meta_path, array_path = _artifact_paths(label)
    if not meta_path.exists() or not array_path.exists():
        raise FileNotFoundError(f"missing artifact for {label}: {meta_path}, {array_path}")
    return json.loads(meta_path.read_text()), np.load(array_path)


def compare_pair(
    left_label: str,
    right_label: str,
    max_calc_diff: float,
    max_js: float,
    report_only: bool,
) -> int:
    left_meta, left_arrays = _load_artifact(left_label)
    right_meta, right_arrays = _load_artifact(right_label)
    try:
        if left_meta["suite_fingerprint"] != right_meta["suite_fingerprint"]:
            raise RuntimeError("suite fingerprints differ; results are not comparable")
        left_records = {record["record_id"]: record for record in left_meta["records"]}
        right_records = {record["record_id"]: record for record in right_meta["records"]}
        if left_records.keys() != right_records.keys():
            raise RuntimeError("artifact record sets differ")

        metrics = []
        for record_id, left_record in left_records.items():
            right_record = right_records[record_id]
            if left_record["selected_ids"] != right_record["selected_ids"]:
                raise RuntimeError(f"{record_id}: selected token sets differ")
            metrics.append(
                _record_metrics(
                    left_arrays[left_record["array_key"]],
                    right_arrays[right_record["array_key"]],
                    left_record,
                )
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in metrics:
            grouped.setdefault(row["case_id"], []).append(row)
        print(f"\n{left_label} -> {right_label}")
        print(
            f"{'case':<26} {'B':>3} {'calc_diff':>11} {'max_abs':>10} "
            f"{'mean_abs':>10} {'JS':>10} {'top1':>7} {'top5':>5} {'flips':>7}"
        )
        failures: list[str] = []
        for case_id, rows in grouped.items():
            calc_diff = max(row["calc_diff"] for row in rows)
            max_abs = max(row["max_abs"] for row in rows)
            mean_abs = max(row["mean_abs"] for row in rows)
            full_rows = [row for row in rows if row["full_vocab"]]
            js = max((row["js_divergence"] for row in full_rows), default=float("nan"))
            top1 = (
                f"{sum(row['top1_same'] for row in full_rows)}/{len(full_rows)}"
                if full_rows
                else "probe"
            )
            top5 = min((row["top5_overlap"] for row in full_rows), default=-1)
            flips = sum(row.get("top1_flip", False) for row in full_rows)
            print(
                f"{case_id:<26} {rows[0]['batch']:>3} {calc_diff:>11.3e} "
                f"{max_abs:>10.4f} {mean_abs:>10.4f} "
                f"{js:>10.3e} {top1:>7} "
                f"{(str(top5) if top5 >= 0 else '-'):>5} {flips:>7}"
            )
            if calc_diff > max_calc_diff:
                failures.append(
                    f"{case_id}: calc_diff {calc_diff:.3e} > {max_calc_diff:.3e}"
                )
            if full_rows and js > max_js:
                failures.append(f"{case_id}: JS {js:.3e} > {max_js:.3e}")

        report = {
            "left": left_label,
            "right": right_label,
            "thresholds": {"max_calc_diff": max_calc_diff, "max_js": max_js},
            "failures": failures,
            "records": metrics,
        }
        report_path = _out_dir() / f"compare.{left_label}_vs_{right_label}.json"
        report_path.write_text(json.dumps(report, indent=2))
        if failures:
            print("FAIL:")
            for failure in failures:
                print(f"  {failure}")
            if not report_only:
                return 1
        print(f"report: {report_path}")
        return 0
    finally:
        left_arrays.close()
        right_arrays.close()


def compare_modes(labels: list[str], report_only: bool) -> int:
    # Real-model tolerances must be agreed from an accepted golden run.  Do
    # not silently turn a guessed tolerance into a correctness contract.
    max_calc_diff = float(os.environ.get("E2E_LOGITS_MAX_CALC_DIFF", "inf"))
    max_js = float(os.environ.get("E2E_LOGITS_MAX_JS", "inf"))
    if math.isinf(max_calc_diff) and math.isinf(max_js):
        print(
            "numeric gates disabled; set E2E_LOGITS_MAX_CALC_DIFF and/or "
            "E2E_LOGITS_MAX_JS to make compare fail on a threshold"
        )
    if len(labels) == 0:
        pairs = [
            ("baseline", candidate)
            for candidate in ("baseline2", "hca", "csa", "mega")
            if all(path.exists() for path in _artifact_paths(candidate))
        ]
    elif len(labels) == 1:
        pairs = [("baseline", labels[0])]
    elif len(labels) == 2:
        pairs = [(labels[0], labels[1])]
    else:
        raise SystemExit("compare accepts zero, one, or two labels")
    if not pairs:
        raise SystemExit(f"no comparable artifacts in {_out_dir()}")
    return max(
        compare_pair(left, right, max_calc_diff, max_js, report_only)
        for left, right in pairs
    )


def self_test() -> None:
    tree = _build_tree_config([10, 11, 12], 0, 1)
    assert tree["prefix_dict"] == {"0": [10], "0_10": [11], "0_10_11": [12]}
    same = _row_metrics(np.array([0.0, 2.0]), np.array([0.0, 2.0]), True)
    assert same["top1_same"] and same["calc_diff"] == 0.0
    near_tie = _row_metrics(
        np.array([1.0, 1.01]), np.array([1.02, 1.0]), True
    )
    assert not near_tie["top1_same"] and near_tie["flip_explained_by_max_abs"]
    flipped = _row_metrics(np.array([0.0, 2.0]), np.array([2.0, 0.0]), True)
    assert not flipped["top1_same"] and flipped["flip_explained_by_max_abs"]
    print("self-test PASS")


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_e2e_forced_logits.py "
            "baseline|baseline2|hca|csa|mega|compare|selftest [labels] [--report-only]"
        )
    action = sys.argv[1]
    report_only = "--report-only" in sys.argv[2:]
    labels = [arg for arg in sys.argv[2:] if arg != "--report-only"]
    if action in MODE_ENVS:
        if labels:
            raise SystemExit(f"{action} does not accept positional arguments")
        run_mode(action)
        return 0
    if action == "compare":
        return compare_modes(labels, report_only)
    if action == "selftest":
        self_test()
        return 0
    raise SystemExit(f"unknown action: {action}")


if __name__ == "__main__":
    raise SystemExit(main())
