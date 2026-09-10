"""Bounded four-thread elastic crossfire with a final quiescent snapshot."""

import json
import math
import random
import threading
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ..contracts import CheckResult, StageHandler, StageOutput


def validate(params, plan):
    from .elastic import _validate

    return _validate(params, plan, set())


def crossfire_validate(params, plan):
    from .elastic import _validate

    # Optional tail_s: extend ONLY the main-loop request emission past the
    # configured mutation storm window (workers still stop at `end`).
    # Absent tail_s (or 0) keeps the behavior identical to the legacy stage.
    p = _validate(
        params,
        plan,
        {"tail_s", "convergence", "mutation_window_s", "traffic", "workers"},
        {"mutation_window_s", "traffic", "workers"},
    )
    if "mutation_window_s" in p and (
        type(p["mutation_window_s"]) not in (int, float)
        or not math.isfinite(p["mutation_window_s"])
        or p["mutation_window_s"] <= 0
    ):
        raise ValueError("mutation_window_s must be positive and finite")
    workers = p["workers"]
    if not isinstance(workers, list) or len(workers) != 4:
        raise ValueError("crossfire requires four explicit worker configurations")
    for worker in workers:
        if not isinstance(worker, dict) or set(worker) != {
            "operation",
            "role",
            "max_operations",
            "interval_s",
            "seed",
        }:
            raise ValueError("worker requires operation, role, count, cadence and seed")
        if worker["operation"] not in {"add", "remove"} or worker["role"] not in {
            "prefill",
            "decode",
        }:
            raise ValueError("unsupported mutation worker")
        if (
            type(worker["max_operations"]) is not int
            or worker["max_operations"] < 1
            or type(worker["seed"]) is not int
        ):
            raise ValueError("worker operation count and seed must be integers")
        if (
            type(worker["interval_s"]) not in (int, float)
            or not math.isfinite(worker["interval_s"])
            or worker["interval_s"] <= 0
        ):
            raise ValueError("worker interval must be positive and finite")
    if sum(w["max_operations"] for w in workers if w["operation"] == "add") > 65:
        raise ValueError("mutation additions exceed reserved port budget")
    traffic = p["traffic"]
    fields = {
        "max_concurrency",
        "interval_s",
        "timeout_s",
        "stream_timeout_s",
        "health_interval_s",
        "input_len",
        "output_len",
    }
    if not isinstance(traffic, dict) or set(traffic) != fields:
        raise ValueError(
            "traffic requires explicit YAML concurrency, cadence, shape and budgets"
        )
    for name, value in traffic.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"traffic.{name} must be positive and finite")
    for name in ("max_concurrency", "input_len", "output_len"):
        if type(traffic[name]) is not int:
            raise ValueError(f"traffic.{name} must be an integer")
    if "tail_s" in p and (
        type(p["tail_s"]) not in (int, float)
        or not math.isfinite(p["tail_s"])
        or p["tail_s"] < 0
    ):
        raise ValueError(f"{plan.path}: tail_s must be a nonnegative number")
    if "convergence" in p:
        c = p["convergence"]
        required = {
            "margin_s",
            "cap_s",
            "post_window_s",
            "min_post_samples",
            "nonbatch_failfast_s",
            "batch_failfast_s",
            "unique_ports",
        }
        if not isinstance(c, dict) or set(c) != required:
            raise ValueError("convergence requires explicit YAML budgets")
        if c["unique_ports"] is not True:
            raise ValueError(
                "convergence requires unique ports for wildcard mock listeners"
            )
        for name, value in c.items():
            if name == "unique_ports":
                continue
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"convergence.{name} must be positive and finite")
        if type(c["min_post_samples"]) is not int:
            raise ValueError("min_post_samples must be an integer")
    return p


def mutation_http(ops, endpoint, deadline, body):
    """Preserve the legacy per-operation client timeout within the stage budget."""
    cap = {"add_engine": 10.0, "remove_engine": 95.0}[endpoint]
    deadline.check()
    request = urllib.request.Request(
        f"http://127.0.0.1:{ops.mock_http_port}/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(cap, deadline.remaining())
    ) as response:
        return json.load(response)


def crossfire(ctx, params, deadline):
    from .elastic import RecordedRequests, completeness
    from .elastic_lifecycle import _master_get

    stop = threading.Event()
    gate = threading.Event()
    done = [threading.Event() for _ in range(4)]
    lock = threading.Lock()
    candidates, operations, health = [], [], []
    addresses = {}
    added_attempts = 0
    errors = []
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    traffic = params["traffic"]
    pool = ThreadPoolExecutor(
        max_workers=traffic["max_concurrency"],
        thread_name_prefix="elastic-crossfire-request",
    )
    pending = set()
    peak_outstanding = 0
    end = ctx.clock() + params["mutation_window_s"]
    # Diagnostic tail observation: mutation workers still stop at `end`
    # (mutation_end); only the main-loop request emission extends to
    # mutation_end + tail_s. Storm-phase = issued before mutation_end.
    mutation_end = end
    convergence = params.get("convergence")
    bound_s = None
    if convergence:
        from .elastic_convergence import convergence_bound

        bound_s = convergence_bound(
            ctx.instance["environment"]["resolved_config"],
            convergence["margin_s"],
            convergence["cap_s"],
        )
    issue_end = mutation_end + params.get("tail_s", 0)
    if convergence:
        issue_end = mutation_end + bound_s + convergence["post_window_s"]
    path = ctx.artifact_dir / f"elastic-crossfire-{time.time_ns()}.json"

    def persist():
        with lock:
            data = dict(
                operations=list(operations),
                health=list(health),
                errors=list(errors),
                mutation_end_s=mutation_end,
                bound_s=bound_s,
                traffic=dict(traffic),
                peak_outstanding=peak_outstanding,
            )
        data.update(
            requests=records.snapshot_records(), workers_done=[e.is_set() for e in done]
        )
        path.write_text(json.dumps(data, indent=2))

    def finish(d):
        stop.set()
        gate.set()
        records.cancel_active("crossfire_cleanup")
        try:
            if pending:
                _, unfinished = wait(pending, timeout=max(0, d.remaining()))
                if unfinished:
                    raise TimeoutError("crossfire request consumers did not terminate")
            pool.shutdown(wait=True)
            for event in done:
                if not event.wait(max(0, d.remaining())):
                    raise TimeoutError("crossfire mutation worker did not terminate")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            persist()

    request_handle = ctx.register_resource("requests", records, cleanup=finish)

    def mutate(index):
        nonlocal added_attempts
        try:
            gate.wait()
            worker = params["workers"][index]
            adder = worker["operation"] == "add"
            cap = worker["max_operations"]
            interval = worker["interval_s"]
            rng = random.Random(worker["seed"])
            for _ in range(cap):
                if stop.is_set() or ctx.clock() >= end:
                    break
                body = dict(role=worker["role"]) if adder else None
                if adder and convergence:
                    # Mock listeners bind wildcard addresses. Reusing a port with
                    # a new advertised IP would let stale-IP traffic reach the new
                    # engine, invalidating the removed-address fail-fast probe.
                    with lock:
                        body["port"] = (
                            ctx.env.base_grpc_port
                            + ctx.env.spec.n_prefill
                            + ctx.env.spec.n_decode
                            + added_attempts
                        )
                        added_attempts += 1
                if not adder:
                    with lock:
                        if candidates:
                            body = dict(port=rng.choice(candidates), mode="graceful")
                if body is not None:
                    entry = dict(
                        worker=index,
                        operation="add" if adder else "remove",
                        request=body,
                        started_s=ctx.clock(),
                    )
                    if convergence and not adder:
                        entry["address"] = addresses.get(body["port"])
                    try:
                        response = mutation_http(
                            ctx.ops,
                            "add_engine" if adder else "remove_engine",
                            deadline,
                            body,
                        )
                        if convergence:
                            entry["ended_s"] = ctx.clock()
                        entry["response"] = response
                        entry["ok"] = (
                            isinstance(response, dict)
                            and response.get("status") == "ok"
                        )
                        if adder and entry["ok"]:
                            port = response.get("port")
                            if type(port) is not int or not 1 <= port <= 65535:
                                raise ValueError("successful add lacks a valid port")
                            if convergence:
                                # Read the exact advertised address after add completion,
                                # before making this port available to remover workers.
                                discovery = json.loads(
                                    ctx.env.discovery_file.read_text()
                                )
                                matches = [
                                    addr
                                    for entries in discovery.values()
                                    if isinstance(entries, list)
                                    for addr in entries
                                    if isinstance(addr, str)
                                    and addr.rsplit(":", 1)[-1] == str(port - 1)
                                ]
                                if len(matches) != 1:
                                    raise ValueError(
                                        "successful add lacks one exact advertised address"
                                    )
                                entry["address"] = (
                                    matches[0].rsplit(":", 1)[0] + ":" + str(port)
                                )
                                addresses[port] = entry["address"]
                            with lock:
                                candidates.append(port)
                    except Exception as exc:
                        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    finally:
                        entry.setdefault("ended_s", ctx.clock())
                        with lock:
                            operations.append(entry)
                            if not adder and body["port"] in candidates:
                                candidates.remove(body["port"])
                stop.wait(max(0, min(interval, end - ctx.clock())))
        except BaseException as exc:
            with lock:
                errors.append(f"worker {index}: {type(exc).__name__}: {exc}")
        finally:
            done[index].set()

    threads = [
        threading.Thread(
            target=mutate, args=(i,), name=f"elastic-crossfire-{i}", daemon=True
        )
        for i in range(4)
    ]
    started = 0
    try:
        for thread in threads:
            thread.start()
            started += 1
        gate.set()

        def keep_issuing():
            nonlocal issue_end
            if convergence:
                with lock:
                    last_remove = max(
                        [mutation_end]
                        + [
                            op["ended_s"]
                            for op in operations
                            if op.get("ok") and op["operation"] == "remove"
                        ]
                    )
                issue_end = last_remove + bound_s + convergence["post_window_s"]
                post_count = sum(
                    r["issued_s"] >= last_remove + bound_s
                    for r in records.snapshot_records()
                )
                return (
                    not all(event.is_set() for event in done)
                    or ctx.clock() < issue_end
                    or post_count < convergence["min_post_samples"]
                )
            return ctx.clock() < issue_end

        next_issue = ctx.clock()
        next_health = next_issue
        while keep_issuing():
            deadline.check()
            completed = {future for future in pending if future.done()}
            for future in completed:
                future.result()
            pending.difference_update(completed)
            if len(pending) >= traffic["max_concurrency"]:
                wait(
                    pending,
                    timeout=min(traffic["interval_s"], deadline.remaining()),
                    return_when=FIRST_COMPLETED,
                )
                continue
            if ctx.clock() < next_issue:
                stop.wait(min(next_issue - ctx.clock(), deadline.remaining()))
                continue
            rid = ctx.ops.next_request_id()
            record = records.issue(rid, ctx.clock)
            records.update(
                record,
                phase=("storm" if record["issued_s"] < mutation_end else "tail"),
            )
            records.update(record, send_due_s=next_issue)
            pending.add(
                pool.submit(
                    records.run,
                    record,
                    dict(
                        input_len=traffic["input_len"],
                        output_len=traffic["output_len"],
                        block_keys=[rid * 100 + 1],
                    ),
                    timeout_s=min(traffic["timeout_s"], deadline.remaining()),
                    stream_timeout_s=traffic["stream_timeout_s"],
                )
            )
            peak_outstanding = max(peak_outstanding, len(pending))
            next_issue += traffic["interval_s"]
            # Sample storm health independently of slow request completion.
            if record.get("phase") == "storm" and ctx.clock() >= next_health:
                next_health = ctx.clock() + traffic["health_interval_s"]
                sample = dict(time_s=ctx.clock())
                try:
                    _master_get(ctx, "rtp_llm/inflight_status", deadline)
                    sample["status"] = 200
                except Exception as exc:
                    sample.update(status=None, error=f"{type(exc).__name__}: {exc}")
                with lock:
                    health.append(sample)
        for future in pending:
            future.result(timeout=max(0, deadline.remaining()))
        pool.shutdown(wait=True)
        # Graceful mutation calls may finish after the mutation admission window.
        # Independent completion events prove they have stopped changing the
        # discovery file before the following consistency measurement.
        for event in done:
            if not event.wait(max(0, deadline.remaining())):
                raise TimeoutError("crossfire mutation worker did not terminate")
        if errors:
            raise RuntimeError("; ".join(errors))
        # Health/success-rate population is the storm phase only: tail records
        # stay out of every existing check numerator/denominator and live on
        # solely as evidence inside the persisted artifact.
        result = completeness(
            [r for r in records.snapshot_records() if r.get("phase") != "tail"]
        )
        if convergence:
            from .elastic_convergence import evaluate_convergence

            result["convergence"] = evaluate_convergence(
                records.snapshot_records(),
                operations,
                mutation_end,
                bound_s,
                batch=ctx.instance["environment"]["resolved_config"]["dispatcher"][
                    "type"
                ]
                == "BATCH",
                min_post_samples=convergence["min_post_samples"],
                nonbatch_failfast_s=convergence["nonbatch_failfast_s"],
                batch_failfast_s=convergence["batch_failfast_s"],
            )
        handle = ctx.register_resource("snapshot", result, historical=True)
        return StageOutput(
            output=dict(
                result=handle,
                requests=request_handle,
                mutation_end_s=mutation_end,
            ),
            checks=[
                CheckResult("workers_finished", "PASS", actual=4),
                CheckResult(
                    "master_http",
                    (
                        "PASS"
                        if health and all(s["status"] == 200 for s in health)
                        else "FAIL"
                    ),
                    evidence=dict(samples=health),
                ),
            ],
            artifacts=[str(path)],
        )
    finally:
        for event in done[started:]:
            event.set()
        gate.set()
        persist()


def discovery(ctx, params, deadline):
    from .elastic import _http

    deadline.check()
    path = ctx.artifact_dir / f"elastic-discovery-{time.time_ns()}.json"
    evidence = {}
    try:
        payload = json.loads(ctx.env.discovery_file.read_text())
        if not isinstance(payload, dict):
            raise ValueError("discovery root must be a mapping")
        counts = {}
        for role in ("prefill", "decode"):
            entries = payload.get(f"mock.{role}.hosts.address")
            if not isinstance(entries, list) or any(
                not isinstance(v, str) for v in entries
            ):
                raise ValueError(f"discovery lacks {role} address list")
            counts[role] = len(entries)
        evidence.update(payload=payload, counts=counts)
    except (OSError, ValueError) as exc:
        evidence["parse_error"] = f"{type(exc).__name__}: {exc}"
        path.write_text(json.dumps(evidence, indent=2))
        return StageOutput(
            checks=[
                CheckResult("parsable", "FAIL", evidence=evidence),
                CheckResult("counts_match", "FAIL", evidence=evidence),
            ],
            artifacts=[str(path)],
        )
    engines = _http(ctx.ops, "snapshot", deadline)["engines"]
    if any(e.get("role") not in ("prefill", "decode") for e in engines):
        raise ValueError("snapshot has unknown role")
    actual = {role: sum(e["role"] == role for e in engines) for role in counts}
    evidence["snapshot"] = engines
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult("parsable", "PASS"),
            CheckResult(
                "counts_match",
                "PASS" if counts == actual else "FAIL",
                actual=actual,
                expected=counts,
                evidence=evidence,
            ),
        ],
        artifacts=[str(path)],
    )


def protocol_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"requests", "method"}, {"requests", "method"})
    plan.reference(p["requests"], "requests")
    if p["method"] not in {"FetchResponse", "GenerateStreamCall"}:
        raise ValueError("crossfire protocol requires a supported stream method")
    return p


def protocol(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["requests"], "requests").snapshot_records()
    # Tail-phase requests (crossfire tail_s) are evidence-only: they stay out
    # of the method check population just like every other existing check.
    admitted = [
        r
        for r in rows
        if r.get("phase") != "tail"
        and r["schedule"]["status"] == "OK"
        and r["prefill_addr"]
    ]
    methods = {str(r["wire_request_id"]): r["stream"]["method"] for r in admitted}
    # Fetch is the client branch selected by Schedule metadata, not proof
    # that every request has already been enqueued on the engine.
    return StageOutput(
        checks=[
            CheckResult(
                "method",
                (
                    "PASS"
                    if methods and all(m == params["method"] for m in methods.values())
                    else "FAIL"
                ),
                actual=methods,
                expected=params["method"],
            )
        ]
    )


def convergence_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"result"}, {"result"})
    plan.reference(p["result"], "snapshot")
    return p


def convergence_verdict(ctx, params, deadline):
    deadline.check()
    evidence = ctx.resource(params["result"], "snapshot")["convergence"]
    path = ctx.artifact_dir / f"elastic-convergence-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(
                "coverage",
                "PASS" if evidence["coverage_ok"] else "FAIL",
                evidence=evidence,
            ),
            CheckResult(
                "bounded_convergence",
                "FAIL" if evidence["late_dead_hits"] else "PASS",
                evidence=evidence,
            ),
            CheckResult(
                "failfast",
                (
                    "FAIL"
                    if evidence["failfast_violations"]
                    else "PASS" if evidence["failfast_samples"] else "SKIP"
                ),
                detail=(
                    "No qualifying failure sample"
                    if not evidence["failfast_samples"]
                    else ""
                ),
                evidence=evidence,
            ),
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler(
        "elastic_convergence_verdict",
        convergence_validate,
        convergence_verdict,
        {},
        checks=frozenset({"coverage", "bounded_convergence", "failfast"}),
    ),
    StageHandler(
        "elastic_crossfire_protocol",
        protocol_validate,
        protocol,
        {},
        checks=frozenset({"method"}),
    ),
    StageHandler(
        "elastic_crossfire",
        crossfire_validate,
        crossfire,
        {"result": "snapshot", "requests": "requests", "mutation_end_s": "number"},
        checks=frozenset({"workers_finished", "master_http"}),
        max_dynamic_additions=65,
    ),
    StageHandler(
        "elastic_discovery_consistency",
        validate,
        discovery,
        {},
        checks=frozenset({"parsable", "counts_match"}),
    ),
]
