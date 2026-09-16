"""Observe a real Schedule/Fetch protocol with deliberately absent client Fetch."""

import copy
import json
import uuid

from ..contracts import CheckResult, StageHandler, StageOutput

CHECKS = frozenset(
    {
        "decode_prepared",
        "fetch_absent",
        "prefill_released_compute",
        "decode_waits_for_fetch",
        "terminal",
        "resources_released",
    }
)


def validate(params, plan):
    if not isinstance(params, dict) or set(params) != {
        "mode",
        "input_len",
        "output_len",
        "observe_s",
    }:
        raise ValueError(
            "client_fetch_probe needs explicit mode, shape and observation window"
        )
    if params["mode"] not in {"late", "missing", "automatic"}:
        raise ValueError("invalid Fetch mode")
    for key in ("input_len", "output_len"):
        if type(params[key]) is not int or params[key] < 2:
            raise ValueError(
                "PD Fetch probe requires positive input and at least two output tokens"
            )
    if (
        type(params["observe_s"]) not in (int, float)
        or not 0 < params["observe_s"] <= 10
    ):
        raise ValueError("observation window must be in (0, 10]")
    if set(plan.profiles) - {"batch-window", "single-batch"}:
        raise ValueError("missing Fetch only exists in BATCH mode")
    return copy.deepcopy(params)


def validate_strict(params, plan):
    result = validate(params, plan)
    if result["mode"] == "automatic":
        raise ValueError("automatic mode requires client_auto_fetch_probe")
    return result


def validate_automatic(params, plan):
    result = validate(params, plan)
    if result["mode"] != "automatic":
        raise ValueError("client_auto_fetch_probe requires automatic mode")
    return result


def execute(ctx, params, deadline):
    import grpc

    ops = ctx.ops
    evidence = dict(mode=params["mode"], env_epoch=ctx.env_epoch, snapshots={})
    checks = []
    path = ctx.artifact_dir / f"client-fetch-{uuid.uuid4().hex}.json"

    def capture(label):
        deadline.check()
        snapshot = ops.snapshot_by_name()
        evidence["snapshots"][label] = snapshot
        return snapshot

    def check(name, actual, expected=True):
        checks.append(
            CheckResult(
                name,
                "PASS" if actual == expected else "FAIL",
                actual=actual,
                expected=expected,
                evidence=evidence,
            )
        )
        return actual == expected

    def wait_snapshot(label, predicate):
        while True:
            snapshot = capture(label)
            if predicate(snapshot):
                return snapshot
            deadline.sleep(0.05)

    def clean(snapshot):
        return all(
            all(
                row[key] == 0
                for key in (
                    "inflight",
                    "prefill_contexts",
                    "decode_waiting_for_kv",
                    "held_blocks",
                    "referenced_blocks",
                    "response_buffers",
                )
            )
            for row in snapshot.values()
        )

    try:
        before = capture("before")
        rid = ops.next_request_id()
        evidence["request_id"] = rid
        response = ops.schedule(
            rid,
            timeout_s=min(15, deadline.remaining()),
            input_len=params["input_len"],
            output_len=params["output_len"],
        )
        if (
            response.code != 200
            or not response.success
            or not response.enqueued_by_master
        ):
            raise RuntimeError(f"BATCH Schedule did not accept request: {response}")
        paddr, daddr = ops.prefill_addr(response), ops.role_addr(response, "DECODE")
        p = next(name for name, row in before.items() if row["grpc_addr"] == paddr)
        d = next(name for name, row in before.items() if row["grpc_addr"] == daddr)
        evidence.update(prefill=p, decode=d)
        admitted = capture("admitted")
        if params["mode"] != "automatic":
            if not check(
                "decode_prepared",
                admitted[d]["decode_waiting_for_kv"] == 1
                and admitted[d]["active_decode_requests"] == 0
                and admitted[p]["inflight"] == 1,
            ):
                checks.extend(
                    CheckResult(
                        name, "ERROR", detail="Decode preparation prerequisite failed"
                    )
                    for name in sorted(CHECKS - {"decode_prepared"})
                )
                return StageOutput({}, checks, [str(path)])
            computed = wait_snapshot("prefill_done", lambda s: s[p]["inflight"] == 0)
            check(
                "prefill_released_compute",
                computed[p]["prefill_contexts"] == 1
                and computed[p]["held_blocks"] + computed[p]["referenced_blocks"] > 0,
            )
            deadline.sleep(params["observe_s"])
            held = capture("no_fetch_window")
            check(
                "decode_waits_for_fetch",
                held[d]["decode_waiting_for_kv"] == 1
                and held[d]["active_decode_requests"] == 0
                and held[d]["completed"] == before[d]["completed"],
            )
        absent = capture("before_any_fetch")
        check(
            "fetch_absent",
            absent[p]["rpc_counts"]["fetch_response"],
            before[p]["rpc_counts"]["fetch_response"],
        )
        stub = ops.pb2_grpc.RpcServiceStub(ops._channel(paddr))
        request = ops.pb2.FetchRequestPB(request_id=rid)
        if params["mode"] == "late":
            frames = list(
                stub.FetchResponse(request, timeout=min(15, deadline.remaining()))
            )
            evidence["frames"] = [str(frame) for frame in frames]
            check(
                "terminal",
                bool(frames)
                and any(frames[-1].flatten_output.finished)
                and not any(frame.HasField("error_info") for frame in frames),
            )
        elif params["mode"] == "missing":
            expired = wait_snapshot("expired", clean)
            terminal = expired[d]["completed"] == before[d]["completed"] and (
                expired[p]["fetch_attach_expirations"]
                == before[p]["fetch_attach_expirations"] + 1
            )
            # Probe only AFTER expiry, proving the context cannot be resurrected.
            try:
                list(stub.FetchResponse(request, timeout=min(5, deadline.remaining())))
                evidence["late_fetch_status"] = "OK"
            except grpc.RpcError as error:
                evidence["late_fetch_status"] = error.code().name
            check("terminal", terminal and evidence["late_fetch_status"] == "NOT_FOUND")
        else:
            completed = wait_snapshot(
                "automatic_done", lambda s: s[d]["completed"] > before[d]["completed"]
            )
            check(
                "terminal",
                completed[d]["completed"] == before[d]["completed"] + 1
                and completed[p]["rpc_counts"]["fetch_response"]
                == before[p]["rpc_counts"]["fetch_response"],
            )
        check("resources_released", clean(wait_snapshot("released", clean)))
        return StageOutput({}, checks, [str(path)])
    finally:
        path.write_text(json.dumps(evidence, indent=2))


HANDLERS = [
    StageHandler("client_fetch_probe", validate_strict, execute, {}, checks=CHECKS),
    StageHandler(
        "client_auto_fetch_probe",
        validate_automatic,
        execute,
        {},
        checks=frozenset({"fetch_absent", "terminal", "resources_released"}),
    ),
]
