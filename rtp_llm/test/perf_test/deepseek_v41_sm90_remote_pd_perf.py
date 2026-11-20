"""Measure existing CP8/DP8 SM90 endpoints using real, explicitly routed KV.

Run this CPU-only client inside the development container after the independent
semantic smoke passes. It does not launch services, acquire GPUs or synthesize
KV. The controller retains both role manifests, GPU telemetry and trace files.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def request(endpoint, path, payload=None, timeout=900):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(req, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        return dict(error_code="HTTP_ERROR", status=error.code,
                    raw_body=error.read().decode(errors="replace"))
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return dict(error_code="INVALID_JSON", raw_body=body)


def routes(endpoint, role, count):
    workers = request(endpoint, "/worker_status", timeout=30)["results"]
    assert len(workers) == count, workers
    assert {int(worker["dp_rank"]) for worker in workers} == set(range(count))
    assert all(worker["alive"] for worker in workers), workers
    result = []
    for worker in sorted(workers, key=lambda item: int(item["dp_rank"])):
        ip, port = worker["address"].rsplit(":", 1)
        result.append(dict(role=role, ip=ip, grpc_port=int(port),
                           http_port=urllib.parse.urlparse(endpoint).port))
    return result


def exact_prompt(endpoint, length):
    """Use the serving tokenizer and fail if its actual length is not exact."""
    prefix = "Continue counting: 1, 2, 3,"
    encode = lambda text: request(endpoint, "/tokenize", {"prompt": text})["token_ids"]
    prefix_ids = encode(prefix)
    assert length >= len(prefix_ids)
    prompt = prefix + " hello" * (length - len(prefix_ids))
    ids = encode(prompt)
    if len(ids) != length:
        # Character search also covers tokenizers with a multi-token filler.
        while len(ids) < length:
            prompt += prompt
            ids = encode(prompt)
        lo, hi = 0, len(prompt)
        while lo <= hi:
            mid = (lo + hi) // 2
            trial = encode(prompt[:mid])
            if len(trial) == length:
                prompt, ids = prompt[:mid], trial
                break
            if len(trial) < length:
                lo = mid + 1
            else:
                hi = mid - 1
    assert len(ids) == length, (length, len(ids))
    return prompt, hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def checked_metrics(raw, length, output_tokens, gamma, decode_route):
    assert raw.get("finished") is True and "error_code" not in raw, raw
    aux = raw["aux_info"]
    assert aux["pd_sep"] is True
    assert aux["input_len"] == length and aux["output_len"] == output_tokens
    assert aux["prefill_total_reuse_len"] == 0, "Perf grid must perform fresh P/D transfer"
    assert len(raw["output_ids"]) == 1 and len(raw["output_ids"][0]) == output_tokens
    assert any(
        route.get("role") == "DECODE"
        and route.get("ip") == decode_route["ip"]
        and int(route.get("grpc_port", -1)) == decode_route["grpc_port"]
        for route in aux.get("role_addrs", [])
    ), aux.get("role_addrs")
    rounds = aux.get("speculative_draft_rounds", 0)
    accepted = aux.get("speculative_accepted_tokens_per_pos", [])
    verified = aux.get("speculative_verified_draft_tokens", 0)
    if gamma:
        assert rounds > 0 and len(accepted) == gamma, aux
        assert all(0 <= count <= rounds for count in accepted), aux
        assert 0 <= sum(accepted) <= verified <= rounds * gamma, aux
        assert verified > 0, "Packed verification must expose its actual denominator"
    else:
        assert rounds == 0 and sum(accepted) == 0 and verified == 0, aux
    ttft = float(aux["first_token_cost_time"])
    tpot = (float(aux["cost_time"]) - ttft) / (output_tokens - 1)
    assert math.isfinite(ttft) and ttft > 0 and math.isfinite(tpot) and tpot > 0
    return dict(ttft_ms=ttft, tpot_ms=tpot, draft_rounds=rounds,
                accepted_tokens=sum(accepted), accepted_per_position=accepted,
                verified_draft_tokens=verified)


def set_backend_log_levels(prefill_endpoints, decode):
    """Frontend /set_log_level only updates rank0; acknowledge every backend."""
    assert len(prefill_endpoints) == len(set(prefill_endpoints)) == 8
    endpoints = [("prefill", rank, endpoint) for rank, endpoint in enumerate(prefill_endpoints)]
    # ServerConfig defines RPC at base+1 and the backend HTTP API at base+5.
    endpoints += [("decode", rank, f"http://{worker['ip']}:{worker['grpc_port'] + 4}")
                  for rank, worker in enumerate(decode)]
    assert len(endpoints) == 16 and len({endpoint for _, _, endpoint in endpoints}) == 16
    receipts = []
    for role, rank, endpoint in endpoints:
        response = request(endpoint, "/set_log_level", {"log_level": "INFO"}, timeout=10)
        assert response == {"status": "ok"}, (role, rank, endpoint, response)
        receipts.append(dict(role=role, rank=rank, endpoint=endpoint, level="INFO", response=response))
    return receipts


def active_decode_snapshot(endpoint):
    """Retain real scheduler IDs/state, excluding waiting and dummy work."""
    # Suppress completed-request history, which is unrelated to the live gate
    # and can otherwise make each eight-rank observation several MB.
    observed = request(endpoint, "/worker_status",
                       {"latest_finished_version": 9223372036854775807}, timeout=10)
    workers = observed["results"]
    assert len(workers) == 8 and all(worker["alive"] for worker in workers)
    active, running = [], []
    for worker in workers:
        for task in worker.get("running_task_info", []):
            running.append(dict(rank=int(worker["dp_rank"]), **task))
            if task.get("phase") == "TASK_PHASE_RUNNING" and not task.get("is_waiting", True):
                active.append(dict(rank=int(worker["dp_rank"]), request_id=task["request_id"],
                                   iterate_count=int(task["iterate_count"]),
                                   input_length=int(task["input_length"]),
                                   backend_status_us=int(worker["status_version"])))
    return dict(observed_unix_ns=time.time_ns(), active=active, running_tasks=running)


def profile_active_batch(args, prompt, length, batch, prefill, decode, label):
    """Arm only after a real rank0 request has started decoding.

    Other EP ranks can perform dummy work. Retain scheduler observations so the
    controller can intersect actual trace timestamps with active request IDs;
    GPU events on their own are not proof of a nonempty decode batch.
    """
    activity_path = args.output_dir / f"{label}-scheduler-activity.jsonl"
    assert not activity_path.exists()

    def observe():
        snapshot = active_decode_snapshot(args.decode_endpoint)
        # Preserve evidence even if no real decode activity can be observed.
        with activity_path.open("a") as writer:
            writer.write(json.dumps(snapshot) + "\n")
        return snapshot

    before = observe()
    assert not before["active"], "Profile requires an exclusive, drained client window"
    observations = [before]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_batch, args, prompt, length, batch, prefill, decode, label)
        deadline = time.monotonic() + 120
        previous_rank0 = set()
        while True:
            snapshot = observe()
            observations.append(snapshot)
            rank0 = {task["request_id"] for task in snapshot["active"] if task["rank"] == 0}
            # RuntimeMeta derives phase from live StreamState, but only updates
            # iterate_count when a task finishes. A running task reports zero.
            # Require the same real RUNNING request in consecutive observations.
            if rank0 & previous_rank0:
                break
            previous_rank0 = rank0
            if future.done():
                future.result()  # Preserve any underlying request failure.
                raise AssertionError("Profile batch finished before real decode activity was observed")
            assert time.monotonic() < deadline, "No active rank0 decode request within profile budget"
            time.sleep(0.05)
        armed_unix_ns = time.time_ns()
        response = request(args.decode_endpoint, "/start_profile", dict(
            gen_timeline=True, trace_name=label, start_step=3,
            num_steps=12, enable_all_rank=True,
        ), timeout=60)
        assert response == {"status": "ok"}, response
        observe_until = time.monotonic() + 3
        while time.monotonic() < observe_until and not future.done():
            observations.append(observe())
            time.sleep(0.05)
        result = future.result()
    return dict(label=label, artifact_status="PENDING_CONTROLLER_VERIFICATION",
                rank_scope="PENDING: intersect trace metadata/timestamps with real scheduler activity",
                armed_unix_ns=armed_unix_ns, active_at_arm=snapshot["active"],
                scheduler_observations=observations, activity_file=str(activity_path), result=result)


def run_batch(args, prompt, length, batch, prefill, decode, label):
    def one(index):
        rank = index % len(decode)
        payload = dict(prompt=prompt, generate_config=dict(
            max_new_tokens=args.output_tokens, min_new_tokens=args.output_tokens,
            top_k=1, top_p=1.0, do_sample=False, random_seed=42,
            reuse_cache=False, can_use_pd_separation=True, force_sp_accept=False,
            return_output_ids=True, aux_info=True,
            role_addrs=[prefill[0], decode[rank]],
        ))
        started = time.perf_counter()
        raw = request(args.prefill_endpoint, "/", payload)
        wall = time.perf_counter() - started
        return dict(rank=rank, wall_s=wall, query=payload, actual=raw)

    count = batch * len(decode)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        responses = list(pool.map(one, range(count)))
    wall = time.perf_counter() - started
    # Persist and check outside the timing window. Long-context raw payloads
    # can be tens of MB per batch; disk writes are not HTTP throughput.
    results = []
    for index, response in enumerate(responses):
        path = args.output_dir / f"{label}-request-{index}.json"
        path.write_text(json.dumps(response, ensure_ascii=False))
    for index, response in enumerate(responses):
        rank = response["rank"]
        metrics = checked_metrics(response["actual"], length, args.output_tokens, args.gamma, decode[rank])
        results.append(dict(rank=rank, wall_s=response["wall_s"],
                            raw_file=f"{label}-request-{index}.json", **metrics))
    accepted = sum(row["accepted_tokens"] for row in results)
    verified = sum(row["verified_draft_tokens"] for row in results)
    rounds = sum(row["draft_rounds"] for row in results)
    return dict(
        wall_s=wall, requests=count, offered_per_rank=batch,
        output_tokens=count * args.output_tokens,
        http_output_tokens_per_s=count * args.output_tokens / wall,
        mean_ttft_ms=statistics.mean(row["ttft_ms"] for row in results),
        mean_tpot_ms=statistics.mean(row["tpot_ms"] for row in results),
        accepted_tokens=accepted, verified_draft_tokens=verified, draft_rounds=rounds,
        accepted_per_verified=None if not verified else accepted / verified,
        accepted_per_round=None if not rounds else accepted / rounds,
        responses=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill-endpoint", required=True)
    parser.add_argument("--decode-endpoint", required=True)
    parser.add_argument("--prefill-backend-http-endpoints", required=True,
                        help="Comma-separated eight verified CP backend HTTP control endpoints, in rank order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--public-sha", required=True)
    parser.add_argument("--internal-sha", required=True)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--lengths", default="32768,65536")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-only", action="store_true",
                        help="Collect a separate active profile without repeating completed formal measurements")
    parser.add_argument("--profile-batch", type=int, help="One separate profile window; defaults to the largest batch")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    lengths = [int(value) for value in args.lengths.split(",")]
    assert min(batches + lengths) > 0 and args.output_tokens > 1
    assert args.rounds > 0 and args.warmups >= 1 and args.gamma >= 0
    assert not args.profile_only or args.profile
    assert all(len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
               for sha in (args.public_sha, args.internal_sha))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", run_id=args.run_id, topology="CP8/DP8 EP8",
                  batch_semantics="offered concurrent requests per decode rank",
                  public_sha=args.public_sha, internal_sha=args.internal_sha,
                  prefill_endpoint=args.prefill_endpoint, decode_endpoint=args.decode_endpoint,
                  gamma=args.gamma,
                  output_tokens=args.output_tokens, warmups=args.warmups, rounds=args.rounds,
                  timing_scope="server TTFT and TPOT; whole HTTP batch throughput",
                  profile_only=args.profile_only,
                  acceptance_denominator="actual speculative_verified_draft_tokens",
                  rows=[], profiles=[])
    destination = args.output_dir / "pd-perf-summary.json"

    def save():
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    save()
    try:
        prefill = routes(args.prefill_endpoint, "PREFILL", 1)
        decode = routes(args.decode_endpoint, "DECODE", 8)
        report.update(prefill_routes=prefill, decode_routes=decode)
        report["backend_log_levels"] = set_backend_log_levels(
            args.prefill_backend_http_endpoints.split(","), decode)
        save()
        for length in ([] if args.profile_only else lengths):
            prompt, token_hash = exact_prompt(args.prefill_endpoint, length)
            for batch in batches:
                row = dict(input_tokens=length, offered_per_rank=batch,
                           input_ids_sha256=token_hash, warmup=[], runs=[])
                report["rows"].append(row)
                for repeat in range(args.warmups + args.rounds):
                    phase = "warmup" if repeat < args.warmups else "runs"
                    label = f"n{length}-b{batch}-{phase}-{repeat}"
                    row[phase].append(run_batch(args, prompt, length, batch, prefill, decode, label))
                    save()
        if args.profile:
            # Keep profiling separate from the formal measurements. Artifact
            # existence/GPU events/flush are verified by the owning controller.
            assert args.output_tokens // (args.gamma + 1) > 15
            length = lengths[0]
            prompt, _ = exact_prompt(args.prefill_endpoint, length)
            batch = args.profile_batch or max(batches)
            assert batch in batches, "Profile must use a measured batch"
            label = f"{args.output_dir.name}-profile-g{args.gamma}-b{batch}-n{length}"
            report["profiles"].append(profile_active_batch(args, prompt, length, batch, prefill, decode, label))
            save()
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()
    print(json.dumps(dict(status=report["status"], rows=len(report["rows"]),
                          summary=str(destination))))


if __name__ == "__main__":
    main()
