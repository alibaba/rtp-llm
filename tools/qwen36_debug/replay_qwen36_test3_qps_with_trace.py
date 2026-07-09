#!/opt/conda310/bin/python
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE_SCRIPT = Path("/home/zhenyun.yzy/ai-search-bench/scripts/replay_qwen36_test3_qps.py")


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-script", default=str(DEFAULT_BASE_SCRIPT))
    parser.add_argument("--trace-prefix", default=f"local_qwen36_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--force-greedy", action="store_true")
    parser.add_argument("--greedy-seed", type=int, default=7)
    args, remaining = parser.parse_known_args()
    return args, [sys.argv[0], *remaining]


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("qwen36_base_replay", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load base replay script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_trace_id(trace_prefix: str, request: dict[str, Any], seq: int) -> str:
    request_id = str(request.get("id") or "unknown")
    source_request_id = str(request.get("source_request_id") or "unknown")
    return f"{trace_prefix}_seq{seq:06d}_{request_id}_src{source_request_id}"


def force_greedy_requests(requests: list[dict[str, Any]], seed: int) -> None:
    for request in requests:
        payload = request.setdefault("payload", {})
        payload["temperature"] = 0.0
        payload["top_p"] = 1.0
        payload["top_k"] = 1
        payload["do_sample"] = False
        payload["seed"] = seed
        payload["frequency_penalty"] = 0.0
        payload["presence_penalty"] = 0.0
        payload["repetition_penalty"] = 1.0
        request["payload_hash"] = f"greedy_{base_stable_hash(payload)}"


def base_stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    import hashlib

    return hashlib.sha256(raw).hexdigest()[:16]


async def send_one_with_trace(base: Any, client: Any, args: argparse.Namespace, request: dict[str, Any], seq: int, trace_prefix: str) -> dict[str, Any]:
    trace_id = make_trace_id(trace_prefix, request, seq)
    traced_request = copy.deepcopy(request)
    traced_request.setdefault("payload", {})["trace_id"] = trace_id
    rec = await base.send_one(client, args, traced_request, seq)
    rec["debug_trace_id"] = trace_id
    return rec


async def run_stress_with_trace(base: Any, args: argparse.Namespace, requests: list[dict[str, Any]], output_dir: Path, trace_prefix: str) -> list[dict[str, Any]]:
    import httpx

    runs_path = output_dir / "runs.jsonl"
    long_path = output_dir / "long_outputs.jsonl"
    runs_path.write_text("", encoding="utf-8")
    long_path.write_text("", encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.max_in_flight)
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    deadline = start + args.duration_sec
    interval = 1.0 / args.qps

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def bounded_send(seq: int, request: dict[str, Any]) -> None:
            async with sem:
                rec = await send_one_with_trace(base, client, args, request, seq, trace_prefix)
            async with lock:
                records.append(rec)
                with runs_path.open("a", encoding="utf-8") as out:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if rec.get("output_chars", 0) >= args.long_char_threshold or rec.get("suspicious"):
                    with long_path.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tasks: set[asyncio.Task[None]] = set()
        seq = 0
        while time.perf_counter() < deadline:
            if args.max_requests and seq >= args.max_requests:
                break
            target = start + seq * interval
            sleep_for = target - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            req = requests[seq % len(requests)]
            task = asyncio.create_task(bounded_send(seq, req))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            seq += 1
            if seq % max(1, int(args.qps * 60)) == 0:
                done = len(records)
                success = sum(1 for r in records if r.get("status") == "success")
                long_count = sum(
                    1
                    for r in records
                    if r.get("output_chars", 0) >= args.long_char_threshold or r.get("suspicious")
                )
                print(
                    f"scheduled={seq} done={done} success={success} long_or_suspicious={long_count} in_flight={len(tasks)}"
                )
        if tasks:
            await asyncio.gather(*tasks)
    return records


async def main_async() -> int:
    wrapper_args, base_argv = parse_wrapper_args()
    base = load_base_module(Path(wrapper_args.base_script).expanduser())

    old_argv = sys.argv
    sys.argv = base_argv
    try:
        args = base.parse_args()
    finally:
        sys.argv = old_argv

    if args.qps <= 0:
        raise SystemExit("--qps must be positive")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else base.BENCH_OUTPUT / f"qwen36_test3_local_qps_trace_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.requests_jsonl:
        requests = base.load_requests(Path(args.requests_jsonl).expanduser())
    else:
        requests = base.materialize_requests(args)
    if wrapper_args.force_greedy:
        force_greedy_requests(requests, wrapper_args.greedy_seed)

    requests_path = (
        Path(args.write_requests_jsonl).expanduser()
        if args.write_requests_jsonl
        else output_dir / "requests.jsonl"
    )
    with requests_path.open("w", encoding="utf-8") as out:
        for row in requests:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_context = vars(args)
    run_context["trace_prefix"] = wrapper_args.trace_prefix
    run_context["base_script"] = str(Path(wrapper_args.base_script).expanduser())
    run_context["force_greedy"] = bool(wrapper_args.force_greedy)
    run_context["greedy_seed"] = wrapper_args.greedy_seed
    (output_dir / "run_context.json").write_text(
        json.dumps(run_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"requests: {len(requests)} -> {requests_path}")
    print(f"output_dir: {output_dir}")
    print(f"trace_prefix: {wrapper_args.trace_prefix}")
    if args.dry_run:
        return 0

    records = await run_stress_with_trace(base, args, requests, output_dir, wrapper_args.trace_prefix)
    base.write_summary(output_dir, args, requests, records)
    print(f"wrote summary: {output_dir / 'summary.json'}")
    print(f"wrote long outputs: {output_dir / 'long_outputs.md'}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
