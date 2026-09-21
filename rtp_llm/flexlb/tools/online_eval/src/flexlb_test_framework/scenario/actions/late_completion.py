"""Inject incomplete reporting only after Master has confirmed a real Decode owner.

The target stays prepared until the missing report is observed and prunes the
confirmed owner. Then client Fetch starts normal execution; the genuine terminal
is released with a fresh cursor. Master code and logging are not modified.
"""

import copy
import json
from pathlib import Path

from ..contracts import CheckResult, StageHandler, StageOutput
from .status_protocol import _http

CHECKS = frozenset({"construction", "scheduler_clean"})


def validate(params, plan):
    if set(params) != {"missing_rounds", "delay_ms", "clean_window_s"}:
        raise ValueError("late completion requires explicit timing parameters")
    for key, low, high in (
        ("missing_rounds", 1, 10),
        ("delay_ms", 1, 10000),
        ("clean_window_s", 30, 30),
    ):
        if type(params[key]) is not int or not low <= params[key] <= high:
            raise ValueError(f"invalid {key}")
    if set(plan.profiles) != {"single-batch"}:
        raise ValueError("late completion has exactly one single-batch instance")
    return copy.deepcopy(params)


def execute(ctx, params, deadline):
    evidence = dict(
        env_epoch=ctx.env_epoch, parameters=params, master_samples=[], events=[]
    )
    artifact = Path(ctx.artifact_dir) / "late-completion.json"
    events_path = ctx.env.run_dir / "engine_events.jsonl"
    checks = []
    stream = None
    rid = ctx.ops.next_request_id()
    evidence["request_id"] = rid
    event_offset = 0
    target_port = None

    def events():
        nonlocal event_offset
        if not events_path.exists():
            return evidence["events"]
        with events_path.open() as reader:
            reader.seek(event_offset)
            while True:
                line = reader.readline()
                if not line or not line.endswith("\n"):
                    break
                event_offset = reader.tell()
                event = json.loads(line)
                if (
                    event.get("event") == "worker_status_delivery_fault"
                    and event.get("port") == target_port
                ):
                    target = next(
                        (t for t in event["targets"] if t["rid"] == rid), None
                    )
                    if target is not None:
                        evidence["events"].append(dict(event, target=target))
        return evidence["events"]

    def master(label):
        _, raw = _http(ctx, "master", "rtp_llm/inflight_status", deadline)
        ds = raw.get("decode_endpoints", [])
        if len(ds) != 1 or type(raw.get("scheduler_inflight")) is not int:
            raise RuntimeError("single Decode and valid scheduler count required")
        d = ds[0]
        if any(
            type(d.get(k)) is not int
            for k in ("confirmed_accepted", "confirmed_running")
        ):
            raise RuntimeError("missing confirmed owner evidence")
        sample = dict(
            label=label,
            time_s=ctx.clock(),
            raw=raw,
            confirmed=d["confirmed_accepted"] + d["confirmed_running"],
        )
        evidence["master_samples"].append(sample)
        return sample

    def wait_for(predicate, label, timeout_s=10):
        end = ctx.clock() + timeout_s
        while ctx.clock() < end:
            deadline.check()
            answer = predicate()
            if answer:
                return answer
            deadline.sleep(0.02)
        raise RuntimeError(f"construction prerequisite missing: {label}")

    def inject(kind, **config):
        _, receipt = _http(
            ctx,
            "mock",
            "inject",
            deadline,
            dict(port=target_port, type=kind, enabled=True, rid=rid, **config),
        )
        evidence.setdefault("injections", []).append(
            dict(type=kind, receipt=receipt, time_s=ctx.clock())
        )

    try:
        response = ctx.ops.schedule(
            rid, timeout_s=min(15, deadline.remaining()), input_len=2048, output_len=2
        )
        if (
            response.code != 200
            or not response.success
            or not response.enqueued_by_master
        ):
            raise RuntimeError(f"BATCH request not accepted: {response}")
        target_port = int(ctx.ops.role_addr(response, "DECODE").rsplit(":", 1)[1])
        evidence["decode_port"] = target_port
        # Establish the real Master owner before arming any fault.
        wait_for(
            lambda: master("confirmed")["confirmed"] == 1, "Master confirmed owner"
        )
        # This traces the subsequent normal report; no task can finish before Fetch.
        inject("status_completion_delay", delay_ms=params["delay_ms"])
        seen = wait_for(
            lambda: next(
                (
                    e
                    for e in events()
                    if e["target"]["running"] and not e["target"]["hidden"]
                ),
                None,
            ),
            "native running report",
        )
        evidence["confirmed_status_version"] = seen["status_version"]
        inject("status_missing_rounds", rounds=params["missing_rounds"])
        absent = wait_for(
            lambda: next(
                (
                    e
                    for e in events()
                    if e["target"]["hidden"]
                    and not e["target"]["running"]
                    and not e["target"]["finished"]
                ),
                None,
            ),
            "post-filter missing report",
        )
        evidence["absent_status_version"] = absent["status_version"]
        wait_for(lambda: master("pruned")["confirmed"] == 0, "confirmed owner removed")
        # A subsequent real report rules out a dead engine or permanent RPC suppression.
        wait_for(
            lambda: next(
                (
                    e
                    for e in events()
                    if e["status_version"] > absent["status_version"]
                    and not e["target"]["hidden"]
                    and e["target"]["running"]
                ),
                None,
            ),
            "running report restored",
        )
        evidence["after_running_restored"] = master("restored")
        stub = ctx.ops.pb2_grpc.RpcServiceStub(
            ctx.ops._channel(ctx.ops.prefill_addr(response))
        )
        stream = stub.FetchResponse(
            ctx.ops.pb2.FetchRequestPB(request_id=rid),
            timeout=min(15, deadline.remaining()),
        )
        frames = list(stream)
        evidence["frames"] = [str(f) for f in frames]
        if (
            not frames
            or not any(frames[-1].flatten_output.finished)
            or any(f.HasField("error_info") for f in frames)
        ):
            raise RuntimeError(
                "client must finish successfully before checking scheduler cleanup"
            )
        delivered = wait_for(
            lambda: next(
                (
                    e
                    for e in events()
                    if len(e["target"]["finished"]) == 1
                    and e["target"]["released_version"] > e["requested_version"]
                    and e["target"]["finished"][0]["error_code"] == 0
                ),
                None,
            ),
            "genuine delayed completion sent with fresh cursor",
        )
        version = delivered["target"]["released_version"]
        wait_for(
            lambda: next(
                (
                    e
                    for e in events()
                    if e["requested_version"] >= version
                    and e["status_version"] > delivered["status_version"]
                ),
                None,
            ),
            "Master acknowledged completion cursor",
        )
        evidence["delivered_version"] = version
        checks.append(CheckResult("construction", "PASS", actual=True, expected=True))
        start = ctx.clock()
        evidence["clean_started_s"] = start
        while True:
            sample = master("clean")
            if (
                sample["raw"]["scheduler_inflight"] == 0
                or ctx.clock() - start >= params["clean_window_s"]
            ):
                break
            deadline.sleep(
                max(0, min(0.5, params["clean_window_s"] - (ctx.clock() - start)))
            )
        evidence["clean_finished_s"] = ctx.clock()
        checks.append(
            CheckResult(
                "scheduler_clean",
                "PASS" if sample["raw"]["scheduler_inflight"] == 0 else "FAIL",
                actual=sample["raw"]["scheduler_inflight"],
                expected=0,
                detail="Real late completion must release scheduler ownership within 30 seconds",
            )
        )
    except Exception as error:
        evidence["error"] = repr(error)
        checks = [
            CheckResult(name, "ERROR", detail=str(error)) for name in sorted(CHECKS)
        ]
    finally:
        if stream is not None and not stream.done():
            stream.cancel()
        events()
        artifact.write_text(json.dumps(evidence, indent=2))
    return StageOutput({}, checks, [str(artifact)])


HANDLERS = [StageHandler("late_completion_probe", validate, execute, {}, checks=CHECKS)]
