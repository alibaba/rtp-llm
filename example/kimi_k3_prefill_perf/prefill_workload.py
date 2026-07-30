#!/usr/bin/env python3
"""Warm and profile the fixed 64K K3 Prefill workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

PERF_PROMPT = "<kimi-k3-accuracy-input-ids>"


def make_input_ids(length: int) -> list[int]:
    return [100 + ((index * 7919 + 17) % 160000) for index in range(length)]


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def run_prefill(base_url: str, ids: list[int], timeout: int) -> tuple[float, list[int]]:
    payload = {
        "prompt": PERF_PROMPT,
        "kimi_k3_accuracy_input_ids": ids,
        "yield_generator": True,
        "generate_config": {
            "max_new_tokens": 1,
            "min_new_tokens": 1,
            "do_sample": False,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 1.0,
            "ignore_eos": True,
            "return_incremental": True,
            "return_logits": False,
            "return_hidden_states": False,
            "return_output_ids": True,
            "return_input_ids": False,
            "aux_info": True,
            "can_use_pd_separation": False,
            "reuse_cache": False,
            "random_seed": 20260722,
            "timeout_ms": timeout * 1000,
            "ttft_timeout_ms": timeout * 1000,
        },
    }
    request = Request(
        base_url.rstrip("/") + "/",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict] = []
    begin = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body.lower() == "[done]":
                continue
            event = json.loads(body)
            if "error_code" in event or "error_code_str" in event:
                raise RuntimeError(event)
            if event.get("output_ids") is not None:
                events.append(event)
    elapsed = time.monotonic() - begin
    if len(events) != 1:
        raise RuntimeError(f"expected one output event, got {len(events)}")
    output_ids = events[0]["output_ids"]
    if (
        isinstance(output_ids, list)
        and len(output_ids) == 1
        and isinstance(output_ids[0], list)
    ):
        output_ids = output_ids[0]
    if not isinstance(output_ids, list):
        raise TypeError(f"unexpected output_ids payload: {output_ids!r}")
    return elapsed, [int(token_id) for token_id in output_ids]


def warmup_converged(samples: list[float]) -> bool:
    if len(samples) < 3:
        return False
    recent = samples[-3:]
    median = statistics.median(recent)
    return all(abs(value - median) <= median * 0.05 for value in recent)


def wait_for_traces(trace_dir: Path, trace_name: str, timeout: int) -> list[str]:
    deadline = time.monotonic() + timeout
    expected = [trace_dir / f"{trace_name}_wr{rank}_1.json" for rank in range(8)]
    while time.monotonic() < deadline:
        if all(path.is_file() and path.stat().st_size > 0 for path in expected):
            return [str(path) for path in expected]
        time.sleep(2)
    missing = [str(path) for path in expected if not path.is_file()]
    raise TimeoutError(f"timed out waiting for traces: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:27188")
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--max-warmups", type=int, default=6)
    parser.add_argument("--backend", choices=("cula", "flash_kda"), default="cula")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.length % 8:
        raise ValueError("Sequence Parallel performance input must be divisible by 8")
    if args.max_warmups < 3:
        raise ValueError("--max-warmups must be at least 3")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    ids = make_input_ids(args.length)
    ids_bytes = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    manifest: dict = {
        "input_length": args.length,
        "input_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "max_new_tokens": 1,
        "ignore_eos": True,
        "reuse_cache": False,
        "warmup_policy": {
            "materialization_runs": 1,
            "minimum_full_warmups": 3,
            "convergence": "last three within median +/-5%",
            "maximum_full_warmups": args.max_warmups,
        },
    }

    print(f"[materialize] length={args.length}", flush=True)
    elapsed, output = run_prefill(args.base_url, ids, args.timeout)
    manifest["materialize_seconds"] = elapsed
    manifest["materialize_output_ids"] = output
    print(f"[materialize] elapsed={elapsed:.6f}s output={output}", flush=True)

    warmups: list[float] = []
    for iteration in range(1, args.max_warmups + 1):
        elapsed, output = run_prefill(args.base_url, ids, args.timeout)
        warmups.append(elapsed)
        print(
            f"[warmup] iteration={iteration} elapsed={elapsed:.6f}s output={output}",
            flush=True,
        )
        if warmup_converged(warmups):
            break
    if not warmup_converged(warmups):
        raise RuntimeError(f"representative 64K warmup did not converge: {warmups}")
    manifest["warmup_seconds"] = warmups
    manifest["warmup_count"] = len(warmups)

    trace_name = f"k3_sp_{args.backend}_mega_prefill_{args.length}"
    profile_response = post_json(
        args.base_url.rstrip("/") + "/start_profile",
        {
            "gen_timeline": True,
            "trace_name": trace_name,
            "start_step": 0,
            "num_steps": 1,
            "enable_all_rank": True,
        },
        60,
    )
    manifest["profile_arm_response"] = profile_response
    print(f"[profile-arm] {profile_response}", flush=True)
    time.sleep(2)

    elapsed, output = run_prefill(args.base_url, ids, args.timeout)
    manifest["profile_seconds"] = elapsed
    manifest["profile_output_ids"] = output
    manifest["trace_name"] = trace_name
    print(f"[profile] elapsed={elapsed:.6f}s output={output}", flush=True)

    manifest["trace_files"] = wait_for_traces(args.trace_dir, trace_name, 600)
    output_path = args.output_dir / "run.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"[done] manifest={output_path}", flush=True)


if __name__ == "__main__":
    main()
