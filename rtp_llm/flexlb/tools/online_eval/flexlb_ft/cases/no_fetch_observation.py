"""Normal Schedule-only mock cohort: no status suppression or TTL injection."""

import json
import time
from dataclasses import replace

from ..context import CaseDef, rid_base
from ..debug_client import DebugClient, DebugUnavailable


def _engines(snapshot):
    engines = snapshot.get("engines")
    if not isinstance(engines, list) or not engines:
        raise DebugUnavailable("missing nonempty mock engines source")
    result = {}
    for engine in engines:
        name = engine["name"]
        if name in result:
            raise DebugUnavailable("duplicate engine identity")
        for value in (
            engine.get("accepted"),
            engine.get("rpc_counts", {}).get("fetch_response"),
        ):
            if type(value) is not int or value < 0:
                raise DebugUnavailable("missing accepted or FetchResponse counter")
        result[name] = engine
    return result


def normal_no_fetch_observation(ctx):
    spec = replace(
        ctx.smoke_spec(),
        label=f"normal_no_fetch_{ctx.profile}",
        master_env={"FLEXLB_DEBUG_ENABLED": "true"},
    )
    env = ctx.env_manager.ensure(spec)
    ops = ctx.engine_ops(env)
    client = DebugClient(f"http://127.0.0.1:{env.master_http_port}")
    artifact = ctx.case_dir("normal_no_fetch_observation") / "evidence.json"
    evidence = dict(
        schema_version=1,
        cohort=[],
        samples=[],
        mode="Schedule_only",
        injections=[],
        limitations=[
            "Mock lifecycle completion is not C++ deferred-slot release.",
            "No evidence of GPU slots, connector KV references, or an exact 600s lifetime.",
            "Master owner rows are observations, not a universal zero-owner assertion.",
        ],
    )
    try:
        baseline = _engines(ops.snapshot())
        if any(
            e["accepted"] or e["rpc_counts"]["fetch_response"]
            for e in baseline.values()
        ):
            raise DebugUnavailable(
                "requires fresh isolated environment before any request"
            )
        evidence["baseline"] = baseline
        first = client.snapshot(include="scheduler,queues,prefill,decode,engine")
        if first.payload["status"] != "ok":
            raise DebugUnavailable("initial Master observation is incomplete")
        instance = first.payload["instanceId"]
        rid = ops.next_request_id(rid_base(ctx, "status"))
        response = ops.schedule(rid, output_len=2, timeout_s=15)
        if (
            response.code != 200
            or not response.success
            or not response.enqueued_by_master
        ):
            raise DebugUnavailable(
                "cohort Schedule was not successfully enqueued by Master"
            )
        selected_addr = ops.prefill_addr(response)
        selected = [
            name for name, e in baseline.items() if e.get("grpc_addr") == selected_addr
        ]
        if len(selected) != 1:
            raise DebugUnavailable(
                "selected Prefill does not map to one baseline engine"
            )
        selected = selected[0]
        evidence["cohort"] = [
            dict(
                request_id=str(rid),
                prefill=selected,
                prefill_addr=selected_addr,
                schedule_code=response.code,
                enqueued_by_master=True,
            )
        ]
        end = time.monotonic() + 20
        completed_at = None
        while time.monotonic() < end:
            current = _engines(ops.snapshot())
            if set(current) != set(baseline):
                raise DebugUnavailable("mock membership changed during frozen cohort")
            deltas = {
                name: e["rpc_counts"]["fetch_response"]
                - baseline[name]["rpc_counts"]["fetch_response"]
                for name, e in current.items()
            }
            if any(delta != 0 for delta in deltas.values()):
                raise DebugUnavailable(
                    "FetchResponse occurred inside the no-Fetch observation window"
                )
            prefill = current[selected]
            lifecycle = prefill.get("request_lifecycle", {}).get(str(rid))
            client.timeout_s = min(5, max(0.001, end - time.monotonic()))
            master = client.snapshot(
                request_id=rid, include="scheduler,queues,prefill,decode,engine"
            )
            if master.payload["instanceId"] != instance:
                raise DebugUnavailable("Master instance changed during frozen cohort")
            evidence["samples"].append(
                dict(
                    captured_mono=time.monotonic(),
                    fetch_delta=deltas,
                    mock=current,
                    master=master.payload,
                )
            )
            completed = (
                prefill["accepted"] - baseline[selected]["accepted"] == 1
                and lifecycle is not None
                and lifecycle.get("end_state") == "completed"
                and lifecycle.get("end_ms", 0) > 0
            )
            if completed and completed_at is None:
                completed_at = time.monotonic()
            if (
                completed
                and time.monotonic() - completed_at >= 2
                and master.payload["status"] == "ok"
            ):
                # Required source completeness is separate from owner values.
                for component in master.payload["components"]:
                    master.component(component)
                evidence["no_fetch_window_passed"] = True
                evidence["prefill_terminal"] = lifecycle
                break
            time.sleep(min(0.2, max(0, end - time.monotonic())))
        else:
            raise DebugUnavailable(
                "nonempty Prefill completion and complete owner observation not established"
            )
        # Recovery is explicitly outside the no-Fetch counter window.
        recovery_rid = ops.next_request_id(rid_base(ctx, "status"))
        _, error = ops.run_one_request(
            recovery_rid, output_len=2, typed_stream_error=True
        )
        evidence["recovery"] = dict(
            request_id=str(recovery_rid), error=str(error) if error else None
        )
        if error is not None:
            return False, f"recovery failed: {error}; evidence={artifact}"
        return (
            True,
            f"rid={rid} Prefill completed with FetchResponse delta=0; recovery passed; evidence={artifact}",
        )
    except Exception as error:
        evidence["error"] = f"{type(error).__name__}: {error}"
        return False, f"ERROR normal no-Fetch observation: {error}; evidence={artifact}"
    finally:
        artifact.write_text(json.dumps(evidence, indent=2), encoding="utf-8")


CASE_DEF = CaseDef(
    "normal_no_fetch_observation",
    "status",
    normal_no_fetch_observation,
    profiles=["batch-window"],
    requires=["enqueue_batch"],
    source="no_fetch_observation.py",
)
