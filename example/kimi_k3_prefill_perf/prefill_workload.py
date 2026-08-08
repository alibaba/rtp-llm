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


def run_prefill(
    base_url: str,
    ids: list[int],
    timeout: int,
    *,
    reuse_cache: bool = False,
) -> tuple[float, list[int]]:
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
            "reuse_cache": reuse_cache,
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


def warmup_converged(
    samples: list[float],
    *,
    minimum: int,
    window: int,
    tolerance: float,
) -> bool:
    if len(samples) < minimum:
        return False
    recent = samples[-window:]
    median = statistics.median(recent)
    return all(abs(value - median) <= median * tolerance for value in recent)


def wait_for_traces(trace_dir: Path, trace_name: str, timeout: int) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        matches = [
            sorted(trace_dir.glob(f"{trace_name}_wr{rank}_*.json"))
            for rank in range(8)
        ]
        if all(
            len(rank_matches) == 1
            and rank_matches[0].is_file()
            and rank_matches[0].stat().st_size > 0
            for rank_matches in matches
        ):
            return [str(rank_matches[0]) for rank_matches in matches]
        time.sleep(2)
    missing = [
        f"{trace_dir}/{trace_name}_wr{rank}_*.json"
        for rank, rank_matches in enumerate(matches)
        if len(rank_matches) != 1
        or not rank_matches[0].is_file()
        or rank_matches[0].stat().st_size == 0
    ]
    raise TimeoutError(f"timed out waiting for traces: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:27188")
    parser.add_argument("--length", type=int, default=65536)
    parser.add_argument(
        "--kda-comm-backend",
        choices=("rs_ag", "a2a"),
        default="rs_ag",
    )
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--backend", choices=("cula", "flash_kda"), default="cula")
    parser.add_argument("--mla-backend", choices=("kernel", "flashmla"), default="flashmla")
    parser.add_argument("--min-warmups", type=int, default=10)
    parser.add_argument("--max-warmups", type=int, default=20)
    parser.add_argument("--stability-window", type=int, default=5)
    parser.add_argument("--stability-percent", type=float, default=3.0)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument(
        "--single-shot",
        action="store_true",
        help="Submit exactly one request; intended for capacity diagnostics, not profiling.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Request engine cache allocation/reuse for chunked capacity experiments.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.length % 8:
        raise ValueError("Sequence Parallel performance input must be divisible by 8")
    if args.min_warmups < 10:
        raise ValueError("--min-warmups must be at least 10")
    if args.max_warmups < args.min_warmups:
        raise ValueError("--max-warmups must be >= --min-warmups")
    if not 3 <= args.stability_window <= args.min_warmups:
        raise ValueError("--stability-window must be in [3, min-warmups]")
    if not 0.0 < args.stability_percent <= 5.0:
        raise ValueError("--stability-percent must be in (0, 5]")
    if args.profile_repeats < 1:
        raise ValueError("--profile-repeats must be positive")
    stability_tolerance = args.stability_percent / 100.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    ids = make_input_ids(args.length)
    ids_bytes = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    manifest: dict = {
        "input_length": args.length,
        "input_ids_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "max_new_tokens": 1,
        "ignore_eos": True,
        "reuse_cache": args.reuse_cache,
        "kda_backend": args.backend,
        "kda_comm_backend": args.kda_comm_backend,
        "mla_backend": args.mla_backend,
        "warmup_policy": {
            "materialization_runs": 1,
            "minimum_full_warmups": args.min_warmups,
            "convergence": (
                f"last {args.stability_window} within median "
                f"+/-{args.stability_percent}%"
            ),
            "maximum_full_warmups": args.max_warmups,
        },
        "profile_repeats": args.profile_repeats,
    }

    if args.single_shot:
        elapsed, output = run_prefill(
            args.base_url, ids, args.timeout, reuse_cache=args.reuse_cache
        )
        print(
            json.dumps(
                {
                    "length": args.length,
                    "elapsed_seconds": elapsed,
                    "output_ids": output,
                    "reuse_cache": args.reuse_cache,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    print(f"[materialize] length={args.length}", flush=True)
    elapsed, output = run_prefill(
        args.base_url, ids, args.timeout, reuse_cache=args.reuse_cache
    )
    manifest["materialize_seconds"] = elapsed
    manifest["materialize_output_ids"] = output
    print(f"[materialize] elapsed={elapsed:.6f}s output={output}", flush=True)

    warmups: list[float] = []
    for iteration in range(1, args.max_warmups + 1):
        elapsed, output = run_prefill(
            args.base_url, ids, args.timeout, reuse_cache=args.reuse_cache
        )
        warmups.append(elapsed)
        print(
            f"[warmup] iteration={iteration} elapsed={elapsed:.6f}s output={output}",
            flush=True,
        )
        if warmup_converged(
            warmups,
            minimum=args.min_warmups,
            window=args.stability_window,
            tolerance=stability_tolerance,
        ):
            break
    if not warmup_converged(
        warmups,
        minimum=args.min_warmups,
        window=args.stability_window,
        tolerance=stability_tolerance,
    ):
        raise RuntimeError(f"representative 64K warmup did not converge: {warmups}")
    manifest["warmup_seconds"] = warmups
    manifest["warmup_count"] = len(warmups)

    trace_name = (
        f"k3_{args.kda_comm_backend}_{args.backend}_{args.mla_backend}_mega_prefill_"
        f"{args.length}_steady"
    )
    # Keep Kineto alive for one profiler warmup plus the measured request.
    # The first profiled request is deliberately excluded: it absorbs
    # profiler startup and per-rank annotation initialization.  Capturing
    # multiple measured 64K requests in one Kineto session can itself fill
    # trace buffers unevenly and make later collectives look imbalanced.
    profile_response = post_json(
        args.base_url.rstrip("/") + "/start_profile",
        {
            "gen_timeline": True,
            "trace_name": trace_name,
            "start_step": 0,
            "num_steps": args.profile_repeats + 1,
            "enable_all_rank": True,
        },
        60,
    )
    print(f"[profile-arm] {profile_response}", flush=True)
    time.sleep(2)

    elapsed, output = run_prefill(
        args.base_url, ids, args.timeout, reuse_cache=args.reuse_cache
    )
    profile_warmup = {
        "seconds": elapsed,
        "output_ids": output,
        "excluded_from_measurement": True,
    }
    print(
        f"[profile-warmup] elapsed={elapsed:.6f}s output={output}",
        flush=True,
    )

    profiles: list[dict] = []
    for repeat in range(1, args.profile_repeats + 1):
        elapsed, output = run_prefill(
            args.base_url, ids, args.timeout, reuse_cache=args.reuse_cache
        )
        print(
            f"[profile] repeat={repeat} elapsed={elapsed:.6f}s output={output}",
            flush=True,
        )
        profiles.append(
            {
                "repeat": repeat,
                "seconds": elapsed,
                "output_ids": output,
            }
        )
    trace_files = wait_for_traces(args.trace_dir, trace_name, 600)
    manifest["profile_trace_name"] = trace_name
    manifest["profile_trace_files"] = trace_files
    manifest["profile_warmup"] = profile_warmup
    manifest["profiles"] = profiles
    output_path = args.output_dir / "run.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"[done] manifest={output_path}", flush=True)


if __name__ == "__main__":
    main()
