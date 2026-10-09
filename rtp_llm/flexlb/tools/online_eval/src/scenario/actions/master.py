"""Owned master process faults, restoration, readiness and ledger cleanup."""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.parameters import validate_fields
from runtime.master_control import require_process as _process, master_json as _master_json


def _target(params):
    params.setdefault("target", "single")
    if params["target"] not in ("single", "A", "B"):
        raise ValueError("master target must be single, A, or B")
    return params


def _layout(params, plan):
    layout = getattr(plan, "environment", {}).get("master_layout", "single")
    if (params["target"] == "single") != (layout == "single"):
        raise ValueError("master target does not match the compiled master layout")


@dataclass
class MasterFault:
    target: str
    mode: str
    process: object
    started_s: float
    restored: bool = False
    started_epoch_s: float = 0.0
    injected: bool = False
    restored_epoch_s: float = 0.0

    def cleanup(self, deadline):
        # Keep the original Popen even after EnvManager removes its registry
        # slot. Never signal a PID that might have been reused after it exited.
        if self.mode == "freeze" and not self.restored and self.process.alive():
            self.process.unfreeze()
            self.restored = True
        if self.mode == "kill":
            self.process.proc.wait(timeout=deadline.remaining())


def _fault_validate(params, plan):
    p = _target(validate_fields(params, plan, {"target", "mode"}, {"mode"}))
    if p["mode"] not in ("kill", "freeze"):
        raise ValueError("master fault mode must be kill or freeze")
    _layout(p, plan)
    return p


def _fault(ctx, params, deadline):
    deadline.check()
    process = _process(ctx, params["target"])
    fault = MasterFault(params["target"], params["mode"], process, ctx.clock())
    fault.started_epoch_s = time.time()
    handle = ctx.register_resource("master_fault", fault, fault.cleanup)
    manager = ctx.backend.manager
    if fault.mode == "freeze":
        process.freeze()
    elif fault.target == "single":
        manager.kill_master9(ctx.env)
    else:
        manager.kill_master9_instance(ctx.env, fault.target)
    fault.injected = True
    if fault.mode == "kill":
        _invalidate_master_channel(ctx, fault.target)
    deadline.check()
    return StageOutput(
        {"fault": handle, "pid": process.pid, "started_s": fault.started_s,
         "epoch_s": fault.started_epoch_s}
    )


def _invalidate_master_channel(ctx, target):
    address = (
        ctx.ops.master_target()
        if target == "single"
        else ctx.backend.manager.master_instance_target(ctx.env, target)
    )
    ctx.ops.invalidate_channel(address)


def _restore_validate(params, plan):
    p = validate_fields(params, plan, {"fault"}, {"fault"})
    plan.reference(p["fault"], "master_fault")
    return p


def _restore(ctx, params, deadline):
    fault = ctx.resource(params["fault"], "master_fault")
    if fault.restored:
        raise ValueError("master fault has already been restored")
    deadline.check()
    if fault.mode == "freeze":
        current = _process(ctx, fault.target)
        if current is not fault.process:
            raise RuntimeError("master process changed while frozen")
        current.unfreeze()
    else:
        # Reap before reusing the lane, retaining the fault resource if start
        # fails; backend teardown owns any partially started replacement.
        fault.process.proc.wait(timeout=deadline.remaining())
        if fault.target == "single":
            current = ctx.backend.manager.start_master(ctx.env)
        else:
            current = ctx.backend.manager.restart_master_instance(ctx.env, fault.target)
    fault.restored = True
    fault.restored_epoch_s = time.time()
    deadline.check()
    same = current.pid == fault.process.pid
    if fault.mode == "kill" and not same:
        _invalidate_master_channel(ctx, fault.target)
    return StageOutput(
        {"pid": current.pid, "restored_s": ctx.clock()},
        [
            CheckResult(
                "process_identity",
                "PASS" if same == (fault.mode == "freeze") else "FAIL",
                actual={"before": fault.process.pid, "after": current.pid},
                expected="same process" if fault.mode == "freeze" else "new process",
            )
        ],
    )


def _ready_validate(params, plan):
    p = _target(
        validate_fields(
            params, plan, {"target", "inflight_zero", "client", "prefill_residual_max"}
        )
    )
    p.setdefault("inflight_zero", True)
    if type(p["inflight_zero"]) is not bool:
        raise ValueError("inflight_zero must be boolean")
    if "client" in p:
        plan.reference(p["client"], "ha_client")
        if (
            not p["inflight_zero"]
            or type(p.get("prefill_residual_max")) is not int
            or p["prefill_residual_max"] < 0
        ):
            raise ValueError(
                "bounded clean requires strict owner zero and prefill bound"
            )
    elif "prefill_residual_max" in p:
        raise ValueError("residual bound requires a live client")
    _layout(p, plan)
    return p


def decode_residual_bound(client):
    # MAX_CONCURRENCY is the client's global cap, shared by its targets.
    value = int(client.flow._overrides["MAX_CONCURRENCY"])
    if value < 1:
        raise ValueError("invalid flow concurrency")
    return value


def owner_clean(count, loads, prefill_max=0, decode_max=0):
    return (
        count == 0
        and loads is not None
        and all(v <= prefill_max for v in loads["prefill"])
        and all(v <= decode_max for v in loads["decode"])
    )


def _endpoint_loads(data):
    loads = {}
    for side in ("prefill", "decode"):
        rows = data[side + "_endpoints"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing endpoint ledger observations")
        values = []
        for row in rows:
            key = "inflight_batches" if side == "prefill" else "total_load"
            if side == "decode" and "total_load" not in row:
                key = "inflight_requests"
            value = row[key]
            if isinstance(value, list) and side == "prefill":
                value = len(value)
            if type(value) is not int or value < 0:
                raise ValueError("invalid endpoint ledger count")
            values.append(value)
        loads[side] = values
    return loads


def _ready(ctx, params, deadline):
    expected = {"PREFILL": ctx.env.spec.n_prefill, "DECODE": ctx.env.spec.n_decode}
    decode_max = (
        decode_residual_bound(ctx.resource(params["client"], "ha_client"))
        if "client" in params
        else 0
    )
    prefill_max = params.get("prefill_residual_max", 0)
    samples = []
    artifact = ctx.artifact_dir / f"master-readiness-{len(ctx._resources)}.json"
    # Endpoint failures/missing fields are errors, never observations of zero.
    # Valid but not-yet-converged samples are retained until the stage deadline.
    try:
        while True:
            deadline.check()
            info = _master_json(
                ctx, params["target"], "/rtp_llm/master/info", deadline, True
            )
            inflight = _master_json(
                ctx, params["target"], "/rtp_llm/inflight_status", deadline
            )
            summary = info["worker_summary"]
            if summary is None:
                summary = {}
            if not isinstance(summary, dict):
                raise ValueError("invalid worker summary")
            count = inflight["scheduler_inflight"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid scheduler inflight count")
            counts = {}
            for role in expected:
                row = summary.get(role)
                if row is None:
                    # The role may be absent between removal and rediscovery.
                    # Preserve unknown; it cannot satisfy readiness.
                    counts[role] = None
                    continue
                if not isinstance(row, dict):
                    raise ValueError("invalid worker role summary")
                values = (row["discovered"], row["alive"])
                if any(type(v) is not int or v < 0 for v in values):
                    raise ValueError("invalid topology counts")
                counts[role] = values
            samples.append(
                dict(
                    time_s=ctx.clock(),
                    topology=counts,
                    scheduler_inflight=count,
                    ready=info.get("ready"),
                )
            )
            converged = info.get("ready") is True and all(
                values == (expected[role], expected[role])
                for role, values in counts.items()
            )
            endpoint_loads = {} if not params["inflight_zero"] else None
            if params["inflight_zero"] and converged:
                endpoint_rows = [
                    inflight[side + "_endpoints"] for side in ("prefill", "decode")
                ]
                if any(not isinstance(rows, list) for rows in endpoint_rows):
                    raise ValueError("invalid endpoint ledger observations")
                if all(endpoint_rows):
                    endpoint_loads = _endpoint_loads(inflight)
            samples[-1]["endpoint_loads"] = endpoint_loads
            samples[-1]["bounds"] = {"prefill": prefill_max, "decode": decode_max}
            clean = not params["inflight_zero"] or (
                owner_clean(count, endpoint_loads, prefill_max, decode_max)
            )
            if converged and clean:
                break
            deadline.sleep(0.5)
    finally:
        artifact.write_text(json.dumps(samples, indent=2) + "\n")
    handle = ctx.register_resource("snapshot", samples, historical=True)
    return StageOutput(
        {"snapshot": handle},
        [
            CheckResult("topology", "PASS", actual=counts, expected=expected),
            CheckResult(
                "inflight",
                "PASS",
                actual=count,
                expected=0 if params["inflight_zero"] else "observed",
            ),
        ],
        [str(artifact)],
    )


def _owner_gate_validate(params, plan):
    p = _target(validate_fields(params, plan, {"target"}))
    _layout(p, plan)
    return p


def _wrap_inflight(ctx, params, deadline):
    """Legacy all-zero owner ledger check, independent of topology readiness."""
    samples = []
    artifact = ctx.artifact_dir / f"master-inflight-{len(ctx.outputs)}.json"
    try:
        while True:
            deadline.check()
            raw = _master_json(
                ctx, params["target"], "/rtp_llm/inflight_status", deadline
            )
            scheduler = raw["scheduler_inflight"]
            if type(scheduler) is not int or scheduler < 0:
                raise ValueError("invalid scheduler inflight count")
            # Keep strict missing-owner validation, but preserve the legacy
            # inflight_requests OR total_load fallback when BOTH fields exist.
            loads = _endpoint_loads(raw)
            decode = []
            for row in raw["decode_endpoints"]:
                for field in ("inflight_requests", "total_load"):
                    if field in row and (type(row[field]) is not int or row[field] < 0):
                        raise ValueError("invalid decode inflight count")
                decode.append(
                    row.get("inflight_requests", 0) or row.get("total_load", 0)
                )
            loads["decode"] = decode
            samples.append(
                dict(
                    time_s=ctx.clock(),
                    scheduler_inflight=scheduler,
                    endpoint_loads=loads,
                    raw=raw,
                )
            )
            if scheduler == 0 and not any(
                value for rows in loads.values() for value in rows
            ):
                break
            deadline.sleep(0.5)
    finally:
        artifact.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", samples, historical=True)},
        [
            CheckResult(
                "inflight", "PASS", actual=samples[-1], expected="all owners zero"
            )
        ],
        [str(artifact)],
    )


HANDLERS = [
    StageHandler(
                "master_fault",
                _fault_validate,
                _fault,
                {"fault": "master_fault", "pid": "integer", "started_s": "number",
                 "epoch_s": "number"},
            ),
    StageHandler(
                "master_restore",
                _restore_validate,
                _restore,
                {"pid": "integer", "restored_s": "number"},
                checks=frozenset({"process_identity"}),
            ),
    StageHandler(
                "master_ready",
                _ready_validate,
                _ready,
                {"snapshot": "snapshot"},
                checks=frozenset({"topology", "inflight"}),
            ),
    StageHandler(
                "master_inflight_clean",
                _owner_gate_validate,
                _wrap_inflight,
                {"snapshot": "snapshot"},
                checks=frozenset({"inflight"}),
            ),
]
