"""Typed mock faults with owned, retryable clearing; acknowledgements are not effects."""

import copy
import json
import uuid

from ..contracts import StageHandler, StageOutput
from .engine_control import ENGINE_NAME, _engines, _http

# Deliberately excludes status-report mutations owned by status_protocol.
FAULT_OPTIONS = {
    **{
        name: {}
        for name in (
            "enqueue_error",
            "generate_error",
            "fetch_error",
            "no_respond",
            "cancel_no_respond",
            "cancel_error",
            "cancel_unexpected_status",
        )
    },
    "enqueue_ack_error_code": {"code": (1, 2**63 - 1)},
    "kv_pressure": {"tokens": (1, 2**63 - 1)},
    "queue_depth": {"depth": (1, 2**31 - 1)},
    "crash_after": {"n": (1, 2**31 - 1)},
    "enqueue_delay": {"delay_ms": (1, 2**63 - 1)},
    "generate_delay": {"delay_ms": (1, 2**63 - 1)},
}


def validate_inject(params, plan):
    if not isinstance(params, dict) or set(params) - {"targets", "type", "options"}:
        raise ValueError("invalid engine_inject fields")
    fault_type = params.get("type")
    if not isinstance(fault_type, str) or fault_type not in FAULT_OPTIONS:
        raise ValueError("unsupported engine fault type")
    targets = params.get("targets")
    if not isinstance(targets, list) or not 1 <= len(targets) <= 32:
        raise ValueError("fault targets must contain 1..32 explicit engines")
    for target in targets:
        if isinstance(target, dict):
            plan.reference(target, "string")
        elif not isinstance(target, str) or not ENGINE_NAME.fullmatch(target):
            raise ValueError("invalid engine target")
    options = params.get("options", {})
    schema = FAULT_OPTIONS[fault_type]
    if not isinstance(options, dict) or set(options) != set(schema):
        raise ValueError("fault options must exactly match the typed schema")
    for name, (minimum, maximum) in schema.items():
        value = options[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"invalid integer fault option {name}")
    return copy.deepcopy(dict(params, options=options))


def validate_clear(params, plan):
    if not isinstance(params, dict) or set(params) != {"fault"}:
        raise ValueError("engine_clear requires only a fault handle")
    plan.reference(params["fault"], "engine_fault")
    return copy.deepcopy(params)


class EngineFault:
    def __init__(self, ctx, targets, fault_type, options, claims):
        self.ctx, self.targets, self.fault_type = ctx, targets, fault_type
        self.ops = ctx.ops
        self.options, self.claims = options, claims
        self.epoch = ctx.env_epoch
        self.pending = set()
        self.path = ctx.artifact_dir / f"engine-fault-{uuid.uuid4().hex}.json"
        self.evidence = dict(
            targets=targets,
            type=fault_type,
            options=options,
            env_epoch=self.epoch,
            control_acknowledged=False,
            effect_verified=False,
            effects_require="separate workload and owner-state assertions",
            injection=[],
            clearing=[],
        )

    def persist(self):
        self.evidence["pending_clear"] = sorted(self.pending)
        self.path.write_text(json.dumps(self.evidence, indent=2))

    def send(self, name, enabled, deadline):
        row = dict(engine=name, enabled=enabled, started_s=self.ctx.clock())
        self.evidence["injection" if enabled else "clearing"].append(row)
        try:
            response = _http(
                self.ops,
                "inject",
                deadline,
                dict(
                    engine=name, type=self.fault_type, enabled=enabled, **self.options
                ),
            )
            row["response"] = response
            if (
                response.get("status") != "ok"
                or response.get("engine") != name
                or response.get("type") != self.fault_type
                or type(response.get("port")) is not int
                or response["port"] != self.ports[name]
            ):
                raise ValueError(
                    "fault control lacks successful target/type acknowledgement"
                )
        except Exception as exc:
            row["error"] = repr(exc)
            raise
        finally:
            row["ended_s"] = self.ctx.clock()
            self.persist()

    def cleanup(self, deadline):
        if self.ctx.env_epoch != self.epoch:
            raise ValueError("cannot clear a fault in a different environment epoch")
        errors = []
        for name in sorted(self.pending):
            try:
                self.send(name, False, deadline)
                self.pending.remove(name)
                self.claims.discard((self.epoch, name, self.fault_type))
            except Exception as exc:
                errors.append(exc)
        self.persist()
        if errors:
            raise errors[0]


def inject(ctx, params, deadline):
    targets = [ctx.resolve(target) for target in params["targets"]]
    if any(
        not isinstance(name, str) or not ENGINE_NAME.fullmatch(name) for name in targets
    ) or len(set(targets)) != len(targets):
        raise ValueError("resolved fault targets must be distinct engine names")
    before = _engines(_http(ctx.ops, "snapshot", deadline), targets)
    ports = {}
    for name, entry in before.items():
        try:
            port = int(entry["grpc_addr"].rsplit(":", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError("engine snapshot has no valid endpoint port") from exc
        if not 1 <= port <= 65535:
            raise ValueError("engine snapshot endpoint port outside TCP range")
        ports[name] = port
    claims = getattr(ctx, "_engine_fault_claims", None)
    if claims is None:
        claims = ctx._engine_fault_claims = set()
    keys = {(ctx.env_epoch, name, params["type"]) for name in targets}
    if claims & keys:
        raise ValueError("overlapping active fault ownership")
    # These four flags are exposed by the Java snapshot. Refuse to take over
    # existing visible faults. Other types rely on the isolated scenario's
    # exclusive controls; there is no claim of reading unexposed configuration.
    if params["type"] in {
        "enqueue_error",
        "generate_error",
        "fetch_error",
        "no_respond",
    }:
        for entry in before.values():
            state = entry.get("inject_config", {}).get(params["type"])
            if state is not False:
                raise ValueError("existing or unobserved fault configuration")
    fault = EngineFault(ctx, targets, params["type"], params["options"], claims)
    fault.ports = ports
    fault.evidence["before"] = before
    handle = ctx.register_resource("engine_fault", fault, fault.cleanup)
    for name in targets:
        # Register ownership before POST: an acknowledgement can be lost after
        # the server applies the mutation. Cleanup must still attempt this name.
        claims.add((ctx.env_epoch, name, params["type"]))
        fault.pending.add(name)
        fault.send(name, True, deadline)
    fault.evidence["control_acknowledged"] = True
    fault.persist()
    return StageOutput({"fault": handle}, artifacts=[str(fault.path)])


def clear(ctx, params, deadline):
    fault = ctx.resource(params["fault"], "engine_fault")
    fault.cleanup(deadline)
    return StageOutput({"cleared": True}, artifacts=[str(fault.path)])


HANDLERS = [
    StageHandler("engine_inject", validate_inject, inject, {"fault": "engine_fault"}),
    StageHandler("engine_clear", validate_clear, clear, {"cleared": "boolean"}),
]
