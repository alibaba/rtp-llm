"""Owned master lifecycle stages; process recovery is distinct from request success."""

from __future__ import annotations

import copy
import json
import urllib.request
from dataclasses import dataclass

from ..contracts import CheckResult, StageHandler, StageOutput


def _params(value, plan, allowed, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError(f"{plan.path}: invalid master parameters")
    return copy.deepcopy(value)


def _target(params):
    params.setdefault("target", "single")
    if params["target"] not in ("single", "A", "B"):
        raise ValueError("master target must be single, A, or B")
    return params


def _process(ctx, target):
    if target == "single":
        if getattr(ctx.env, "masters", {}):
            raise ValueError("single target cannot address a dual-master environment")
        proc = ctx.env.master
    else:
        if set(getattr(ctx.env, "master_specs", {})) != {"A", "B"}:
            raise ValueError(
                "A/B target requires an actual dual-standalone environment"
            )
        proc = ctx.env.masters.get(target)
    if proc is None or not proc.alive():
        raise RuntimeError(f"owned master {target} is not running")
    return proc


@dataclass
class MasterFault:
    target: str
    mode: str
    process: object
    started_s: float
    restored: bool = False

    def cleanup(self, deadline):
        # Keep the original Popen even after EnvManager removes its registry
        # slot. Never signal a PID that might have been reused after it exited.
        if self.mode == "freeze" and not self.restored and self.process.alive():
            self.process.unfreeze()
            self.restored = True
        if self.mode == "kill":
            self.process.proc.wait(timeout=deadline.remaining())


def _fault_validate(params, plan):
    p = _target(_params(params, plan, {"target", "mode"}, {"mode"}))
    if p["mode"] not in ("kill", "freeze"):
        raise ValueError("master fault mode must be kill or freeze")
    return p


def _fault(ctx, params, deadline):
    deadline.check()
    process = _process(ctx, params["target"])
    fault = MasterFault(params["target"], params["mode"], process, ctx.clock())
    handle = ctx.register_resource("master_fault", fault, fault.cleanup)
    manager = ctx.backend.manager
    if fault.mode == "freeze":
        process.freeze()
    elif fault.target == "single":
        manager.kill_master9(ctx.env)
    else:
        manager.kill_master9_instance(ctx.env, fault.target)
    deadline.check()
    return StageOutput(
        {"fault": handle, "pid": process.pid, "started_s": fault.started_s}
    )


def _restore_validate(params, plan):
    p = _params(params, plan, {"fault"}, {"fault"})
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
    deadline.check()
    same = current.pid == fault.process.pid
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


def _master_json(ctx, target, path, deadline, post=False):
    _process(ctx, target)
    port = (
        ctx.env.master_http_port
        if target == "single"
        else ctx.env.master_specs[target].http_port
    )
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=b"{}" if post else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(
        request, timeout=min(5, deadline.remaining())
    ) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("master response is not a JSON object")
    return data


def _ready_validate(params, plan):
    p = _target(_params(params, plan, {"target", "inflight_zero"}))
    p.setdefault("inflight_zero", True)
    if type(p["inflight_zero"]) is not bool:
        raise ValueError("inflight_zero must be boolean")
    return p


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
            count = inflight["scheduler_inflight"]
            if type(count) is not int or count < 0:
                raise ValueError("invalid scheduler inflight count")
            counts = {}
            for role in expected:
                row = summary[role]
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
            endpoint_loads = (
                _endpoint_loads(inflight) if params["inflight_zero"] else {}
            )
            samples[-1]["endpoint_loads"] = endpoint_loads
            clean = not params["inflight_zero"] or (
                count == 0
                and not any(v for values in endpoint_loads.values() for v in values)
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


HANDLERS = [
    StageHandler(
        "master_fault",
        _fault_validate,
        _fault,
        {"fault": "master_fault", "pid": "integer", "started_s": "number"},
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
]
