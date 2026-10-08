#!/usr/bin/env python3
"""Capture one 64K/B32-per-owner modeling case after external serving warmup.

This client records requests and profiler acknowledgements. It never treats
requested concurrency as proof of actual batch or HTTP duration as GPU timing.
Run on the Decode host so the trace directory can be checked locally.
"""
import argparse
import gzip
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

FIXTURE_SHA = "291c1a008a064a73afda5fbb7d7daab97aaa64e656b2d00ab44610ed73db7460"


def post(url, payload, timeout=30):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def post_json(url, payload, timeout=30):
    status, raw = post(url, payload, timeout)
    data = json.loads(raw)
    if status != 200 or "error" in data:
        raise RuntimeError(f"{url}: HTTP {status}: {raw[:300]!r}")
    return data


def owner_rows(statuses, tp):
    """Deduplicate frontend broadcasts, requiring all expected DP owners."""
    owners = {}
    for direct_owner, status in enumerate(statuses):
        if "error" in status:
            raise RuntimeError(f'worker_status failed: {status["error"]}')
        for row in status.get("results", [status]):
            if not row.get("alive") or int(row.get("tp_size", 0)) != tp:
                raise RuntimeError("worker_status topology or health mismatch")
            if "dp_rank" in row:
                owner = int(row["dp_rank"])
            elif "results" not in status:
                # Fixed feat has no WorkerStatus.dp_rank field. Its request
                # entries have dp_rank, and each direct URL is explicitly an
                # owner TP-root endpoint. Empty queue snapshots use that route.
                task_owners = {
                    int(t["dp_rank"])
                    for t in row.get("running_task_info", [])
                    if "dp_rank" in t
                }
                if len(task_owners) > 1:
                    raise RuntimeError("Direct owner endpoint reported mixed DP owners")
                owner = next(iter(task_owners)) if task_owners else direct_owner
            else:
                raise RuntimeError("DP broadcast status has no owner identity")
            if int(row.get("dp_size", 0)) != 8 // tp:
                raise RuntimeError("worker_status DP size mismatch")
            # The same backend may be reported by both HTTP frontends. Snapshots
            # can advance between reads; the most recent response wins.
            owners[owner] = row
    if set(owners) != set(range(8 // tp)):
        raise RuntimeError("worker_status did not report every Decode owner")
    return owners


def full_batch_ready(owners, input_tokens=65536):
    for row in owners.values():
        tasks = row.get("running_task_info", [])
        if len(tasks) != 32:
            return False
        if any(
            task.get("is_waiting", False)
            or int(task.get("input_length", -1)) != input_tokens
            for task in tasks
        ):
            return False
    return True


def arm_profile(decode_urls, trace_name, steps, request=post_json, start_step=8):
    # StartProfile(enable_all_rank) fans out to one TP group, not all DP owners.
    # Invoke every owner before accepting the profiler acknowledgement set.
    payload = {
        "trace_name": trace_name,
        "start_step": start_step,
        "num_steps": steps,
        "enable_all_rank": True,
    }
    with ThreadPoolExecutor(max_workers=len(decode_urls)) as pool:
        results = list(
            pool.map(lambda url: request(url + "/start_profile", payload), decode_urls)
        )
    if any(result.get("status") != "ok" for result in results):
        raise RuntimeError("Not every Decode owner acknowledged profiling")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument(
        "--decode-urls",
        nargs="+",
        required=True,
        help="DP1/TP8 performance owner0 URL; DP2/TP4 is correctness-only",
    )
    parser.add_argument("--tp", type=int, choices=(8,), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=40)
    parser.add_argument("--profile-start-step", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    urls = [url.rstrip("/") for url in args.decode_urls]
    if len(urls) != 8 // args.tp or len(set(urls)) != len(urls):
        parser.error("One distinct URL required per Decode TP owner")
    if (
        args.warmup_batches < 10
        or args.profile_steps < 32
        or args.output_tokens < 512
        or args.profile_start_step < 0
    ):
        parser.error(
            "Need >=10 warmup batches, >=32 profiling steps and >=512 output tokens"
        )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,55}", args.trace_prefix):
        parser.error("trace prefix must contain 1..55 safe characters")
    if args.output_dir.exists() or not args.trace_dir.is_dir():
        parser.error("Output must be new and existing trace directory must be local")
    fixture = json.load(gzip.open(args.fixture, "rt"))
    ids = fixture["input_ids"]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    if len(ids) != 65536 or fixture["input_len"] != len(ids) or digest != FIXTURE_SHA:
        parser.error("Expected the approved exact 65536-token long-output fixture")
    prefill_url = args.prefill_url.rstrip("/")
    decode = [urlsplit(url) for url in urls]
    if any(
        not u.hostname or not u.port or u.hostname in ("127.0.0.1", "localhost", "::1")
        for u in decode
    ):
        parser.error("PD Decode URLs need explicit reachable host and port")
    args.output_dir.mkdir(parents=True)
    summary = {
        "input_tokens": len(ids),
        "input_ids_sha256": digest,
        "actual_batch_required_per_owner": 32,
        "tp": args.tp,
        "dp": 8 // args.tp,
        "warmup_batches_per_window_required": args.warmup_batches,
        "output_tokens": args.output_tokens,
        "profile_start_step": args.profile_start_step,
        "prefill_url": prefill_url,
        "decode_urls": urls,
        "groups": [],
        "windows": [],
        "timing_note": "HTTP timings are warmup diagnostics only; use all-rank Graph analysis for acceptance",
        "drain_boundary": "Serving queues drained between independent batches; no standalone CUDA barrier claim",
        "batch_evidence": "worker_status confirms 32 enqueued requests per owner; actual modeling B32 must be confirmed in every rank trace. "
        "Fixed feat does not refresh live iterate_count, so it cannot be a readiness gate.",
    }

    def save():
        (args.output_dir / "client-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    def snapshot():
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            statuses = list(
                pool.map(lambda url: post_json(url + "/worker_status", {}), urls)
            )
        return owner_rows(statuses, args.tp)

    def drain():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            decode_status = snapshot()
            prefill_status = post_json(prefill_url + "/worker_status", {})
            p_rows = prefill_status.get("results", [prefill_status])
            if not p_rows or any(not row.get("alive") for row in p_rows):
                raise RuntimeError("Prefill status is unavailable")
            if all(
                not row.get("running_task_info", []) for row in decode_status.values()
            ) and all(not row.get("running_task_info", []) for row in p_rows):
                return {"decode": decode_status, "prefill": prefill_status}
            time.sleep(0.2)
        raise RuntimeError("Serving queues did not drain")

    def request_group(label, profile_name=None, per_owner=32):
        count = per_owner * len(urls)
        barrier = threading.Barrier(count)

        def one(index):
            owner, local = divmod(index, per_owner)
            route = decode[owner]
            payload = {
                "model": "kimi-k3",
                "messages": fixture["messages"],
                "max_tokens": args.output_tokens,
                "temperature": 0,
                "top_k": 1,
                "top_p": 0.95,
                "seed": 0,
                "stream": False,
                "debug_info": True,
                "enable_thinking": False,
                "extra_configs": {
                    "ignore_eos": True,
                    "reuse_cache": True,
                    "role_addrs": [
                        {
                            "role": "DECODE",
                            "ip": route.hostname,
                            "http_port": route.port,
                            "grpc_port": route.port + 1,
                        }
                    ],
                },
            }
            barrier.wait(timeout=60)
            start = time.monotonic()
            status, raw = post(
                prefill_url + "/v1/chat/completions", payload, args.timeout
            )
            path = args.output_dir / f"{label}-o{owner}-r{local:02d}.json.gz"
            with gzip.open(path, "wb") as output:
                output.write(raw)
            response = json.loads(raw)
            if (
                status != 200
                or "error" in response
                or response.get("debug_info", {}).get("input_ids") != ids
                or response.get("aux_info", {}).get("input_len") != len(ids)
                or response.get("aux_info", {}).get("pd_sep") is not True
                or response.get("usage", {}).get("completion_tokens")
                != args.output_tokens
            ):
                raise RuntimeError(
                    f"{label}/owner{owner}/request{local}: HTTP, PD, token or output check failed"
                )
            return {
                "owner": owner,
                "request": local,
                "http_status": status,
                "elapsed_s_diagnostic_only": time.monotonic() - start,
                "response_sha256": hashlib.sha256(raw).hexdigest(),
                "response": path.name,
                "input_tokens": len(ids),
                "output_tokens": args.output_tokens,
            }

        ready = None
        ack = None
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(one, index) for index in range(count)]
            if per_owner == 32:
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    if any(
                        future.done() and future.exception() is not None
                        for future in futures
                    ):
                        raise RuntimeError(
                            "Request failed before the full B32 warmup/profiling boundary"
                        )
                    if any(future.done() for future in futures):
                        raise RuntimeError(
                            "A request finished before the full B32 boundary"
                        )
                    state = snapshot()
                    if full_batch_ready(state):
                        ready = state
                        if profile_name:
                            ack = arm_profile(
                                urls,
                                profile_name,
                                args.profile_steps,
                                start_step=args.profile_start_step,
                            )
                        break
                    time.sleep(0.05)
                if ready is None:
                    raise RuntimeError("Did not observe real B32 on every Decode owner")
            records = [future.result() for future in futures]
        group = {"label": label, "requests": records, "queues_after": drain()}
        if ready is not None:
            group["ready_snapshot"] = ready
        if ack is not None:
            group.update(
                {
                    "profile_name": profile_name,
                    "ready_snapshot": ready,
                    "profile_ack": ack,
                }
            )
        summary["groups"].append(group)
        save()
        print(json.dumps({"completed": label, "requests": count}), flush=True)
        return group

    save()
    summary["initial_queues"] = drain()
    # Seed the Prefill Memory Cache before concurrent arrivals. This single
    # request does not replace any of the required same-shape B32 warmups.
    request_group("prefill-cache-seed", per_owner=1)
    request_group("materialize")
    for window in range(1, 4):
        warmup_labels = []
        for iteration in range(args.warmup_batches):
            label = f"w{window:02d}-warmup-{iteration+1:02d}"
            request_group(label)
            warmup_labels.append(label)
        trace_name = f"{args.trace_prefix}_w{window:02d}"
        if list(args.trace_dir.glob(trace_name + "_wr*.json")):
            raise RuntimeError("Trace name already exists; refusing to mix sessions")
        request_group(f"w{window:02d}-profile", trace_name)
        # Profiling export is asynchronous. Do not arm the next session until
        # every rank file is stable; the analyzer also parses JSON and SHA-checks.
        deadline = time.monotonic() + 120
        previous = None
        stable = 0
        while time.monotonic() < deadline:
            paths = list(args.trace_dir.glob(trace_name + "_wr*.json"))
            by_rank = {}
            for path in paths:
                match = re.search(r"_wr(\d+)_\d+\.json$", path.name)
                if match:
                    rank = int(match.group(1))
                    if rank in by_rank:
                        raise RuntimeError("Duplicate trace file for a rank")
                    by_rank[rank] = path
            state = {rank: path.stat().st_size for rank, path in by_rank.items()}
            stable = (
                stable + 1 if state == previous and set(state) == set(range(8)) else 0
            )
            previous = state
            if stable >= 3 and all(state.values()):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Eight complete rank traces were not exported")
        traces = []
        for rank, path in sorted(by_rank.items()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1048576), b""):
                    digest.update(chunk)
            traces.append(
                {
                    "rank": rank,
                    "path": str(path.resolve()),
                    "sha256": digest.hexdigest(),
                }
            )
        summary["windows"].append(
            {
                "window": window,
                "warmup_completed_batches": len(warmup_labels),
                "warmup_group_labels": warmup_labels,
                "traces": traces,
            }
        )
        save()
    print(
        "Captured three windows; raw traces still require modeling and configuration audits",
        flush=True,
    )


if __name__ == "__main__":
    main()
