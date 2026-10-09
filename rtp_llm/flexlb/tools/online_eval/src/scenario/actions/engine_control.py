"""Explicit, deadline-bounded mock engine controls shared by scenario families."""

import copy
import json
import math
import uuid

from scenario.contracts import StageHandler, StageOutput
from runtime.mock_control import ENGINE_NAME, control_json, select_engines

PERF_FIELDS = {
    "prefill_fixed_ms",
    "decode_scale",
    "max_waiting_batches",
    "max_prefill_concurrency",
}


def validate(params, plan):
    if not isinstance(params, dict) or set(params) - {"operation", "targets", "perf"}:
        raise ValueError(f"{plan.path}: invalid engine_control fields")
    if params.get("operation") not in {"set_perf", "stop", "start"}:
        raise ValueError(f"{plan.path}: unsupported engine operation")
    targets = params.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 32:
        raise ValueError(f"{plan.path}: targets must contain 1..32 explicit engines")
    for target in targets:
        if isinstance(target, dict):
            plan.reference(target, "string")
        elif not isinstance(target, str) or not ENGINE_NAME.fullmatch(target):
            raise ValueError(f"{plan.path}: invalid engine target")
    perf = params.get("perf", {})
    if params["operation"] != "set_perf":
        if "perf" in params:
            raise ValueError(f"{plan.path}: perf only belongs to set_perf")
    elif not isinstance(perf, dict) or not perf or set(perf) - PERF_FIELDS:
        raise ValueError(f"{plan.path}: unsupported or empty perf fields")
    for key, value in perf.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{plan.path}: perf values must be finite and nonnegative")
        if (
            key in {"max_waiting_batches", "max_prefill_concurrency"}
            and type(value) is not int
        ):
            raise ValueError(
                f"{plan.path}: concurrency and batch caps must be integers"
            )
        if key == "max_prefill_concurrency" and value < 1:
            raise ValueError(f"{plan.path}: prefill concurrency must be positive")
    return copy.deepcopy(params)


def execute(ctx, params, deadline):
    targets = [ctx.resolve(target) for target in params["targets"]]
    if any(
        not isinstance(name, str) or not ENGINE_NAME.fullmatch(name) for name in targets
    ) or len(set(targets)) != len(targets):
        raise ValueError("resolved engine targets must be distinct valid names")
    operation = params["operation"]
    evidence = dict(
        operation=operation,
        targets=targets,
        env_epoch=ctx.env_epoch,
        endpoint_generation=None,
        started_s=ctx.clock(),
        responses=[],
        complete=False,
    )
    path = ctx.artifact_dir / f"engine-control-{uuid.uuid4().hex}.json"
    try:
        evidence["before"] = select_engines(control_json(ctx.ops, "snapshot", deadline), targets)
        for name in targets:
            body = dict(engine=name, **params.get("perf", {}))
            response = control_json(
                ctx.ops,
                "set_perf" if operation == "set_perf" else operation + "_engine",
                deadline,
                body,
            )
            evidence["responses"].append(response)
            if (
                response.get("status") != "ok"
                or response.get("engine") != name
                or type(response.get("port")) is not int
            ):
                raise ValueError(
                    f"{operation}({name}) lacks successful target acknowledgement"
                )
        evidence["after"] = select_engines(control_json(ctx.ops, "snapshot", deadline), targets)
        for name in targets:
            before, after = evidence["before"][name], evidence["after"][name]
            if (
                before["role"] != after["role"]
                or before["grpc_addr"] != after["grpc_addr"]
            ):
                raise ValueError(
                    f"{name}: endpoint identity changed during engine control"
                )
            if operation in {"stop", "start"} and after["stopped"] != (
                operation == "stop"
            ):
                raise ValueError(
                    f"{name}: acknowledged {operation} without matching snapshot state"
                )
        evidence.update(
            complete=True,
            sample_count=len(targets),
            min_samples=len(targets),
            requested_perf=params.get("perf"),
            effect_verified=operation != "set_perf",
            perf_verification=(
                "control acknowledgement only; workload latency requires separate observation"
                if operation == "set_perf"
                else None
            ),
        )
    except Exception as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2))
    handle = ctx.register_resource("snapshot", evidence, historical=True)
    return StageOutput(output={"snapshot": handle}, artifacts=[str(path)])


HANDLERS = [StageHandler("engine_control", validate, execute, {"snapshot": "snapshot"})]
