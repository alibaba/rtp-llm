"""Bounded four-thread elastic crossfire with a final quiescent snapshot."""

import json
import random
import threading
import time
import urllib.request

from ..contracts import CheckResult, StageHandler, StageOutput


def validate(params, plan):
    from .elastic import _validate

    return _validate(params, plan, set())


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
    errors = []
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    end = ctx.clock() + 10
    path = ctx.artifact_dir / f"elastic-crossfire-{time.time_ns()}.json"

    def persist():
        with lock:
            data = dict(
                operations=list(operations), health=list(health), errors=list(errors)
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
            for event in done:
                if not event.wait(max(0, d.remaining())):
                    raise TimeoutError("crossfire mutation worker did not terminate")
        finally:
            persist()

    ctx.register_resource("requests", records, cleanup=finish)

    def mutate(index):
        try:
            gate.wait()
            adder = index < 2
            worker = index % 2
            # At most ceil(10/.25)+ceil(10/.40)=65 add attempts. Failed
            # attempts also consume the budget; the mock may allocate a port
            # before a failing response reaches this caller.
            cap = (40 if worker == 0 else 25) if adder else (25 if worker == 0 else 19)
            interval = (0.25 if adder else 0.4) + 0.15 * worker
            rng = random.Random(worker * 977)
            for _ in range(cap):
                if stop.is_set() or ctx.clock() >= end:
                    break
                body = (
                    dict(role="prefill" if worker == 0 else "decode") if adder else None
                )
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
                    try:
                        response = mutation_http(
                            ctx.ops,
                            "add_engine" if adder else "remove_engine",
                            deadline,
                            body,
                        )
                        entry["response"] = response
                        entry["ok"] = (
                            isinstance(response, dict)
                            and response.get("status") == "ok"
                        )
                        if adder and entry["ok"]:
                            port = response.get("port")
                            if type(port) is not int or not 1 <= port <= 65535:
                                raise ValueError("successful add lacks a valid port")
                            with lock:
                                candidates.append(port)
                    except Exception as exc:
                        entry.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    finally:
                        entry["ended_s"] = ctx.clock()
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
        while ctx.clock() < end:
            deadline.check()
            rid = ctx.ops.next_request_id()
            record = records.issue(rid, ctx.clock)
            records.run(
                record,
                dict(output_len=2, block_keys=[rid * 100 + 1]),
                timeout_s=min(40, deadline.remaining()),
                stream_timeout_s=10,
            )
            sample = dict(time_s=ctx.clock())
            try:
                _master_get(ctx, "rtp_llm/inflight_status", deadline)
                sample["status"] = 200
            except Exception as exc:
                sample.update(status=None, error=f"{type(exc).__name__}: {exc}")
            with lock:
                health.append(sample)
            stop.wait(max(0, min(1, end - ctx.clock())))
        # Graceful mutation calls may finish after the 10s admission window.
        # Independent completion events prove they have stopped changing the
        # discovery file before the following consistency measurement.
        for event in done:
            if not event.wait(max(0, deadline.remaining())):
                raise TimeoutError("crossfire mutation worker did not terminate")
        if errors:
            raise RuntimeError("; ".join(errors))
        result = completeness(records.snapshot_records())
        handle = ctx.register_resource("snapshot", result, historical=True)
        return StageOutput(
            output=dict(result=handle),
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


HANDLERS = [
    StageHandler(
        "elastic_crossfire",
        validate,
        crossfire,
        {"result": "snapshot"},
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
