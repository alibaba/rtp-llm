"""Execute a compiled instance through stages, owned resources and finalization."""

import copy
import time
from runtime.deadline import Deadline, StageTimeout, interruptible
from scenario.context import RuntimeContext
from scenario.stage_execution import StageExecutor
from scenario.finalization import RunFinalizer


def execute_instance(
    instance,
    backend,
    handlers=None,
    artifact_dir=".",
    clock=time.monotonic,
    sleeper=time.sleep,
    cancelled=None,
    enforce_deadlines=False,
    _policy=None,
):
    """Execute a compiled plan. Adapters must enforce deadlines in actual work."""
    handlers = dict(handlers or {})
    ctx = RuntimeContext(
        instance, backend, artifact_dir, clock, sleeper, enforce_deadlines
    )
    ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
    started = clock()
    end = started + instance["execution"]["timeout_s"]
    ctx.instance_deadline_s = end
    rows, terminal_status, primary_error, finding_failures, finding_passes, cleanup = StageExecutor().run(
        ctx, handlers, started, end, cancelled, _policy)
    result = {
        key: instance[key]
        for key in (
            "id",
            "scenario_id",
            "variant_id",
            "profile",
            "category",
            "source",
        )
    }
    result.update(
        status=terminal_status,
        grade=instance.get("grade", "normal"),
        effective_axes=instance.get("effective_axes"),
        effective_capabilities=instance.get("effective_capabilities"),
        stages=rows,
        cleanup=cleanup,
        error=primary_error,
        duration_ms=int((clock() - started) * 1000),
        finding_confirmed=finding_failures,
        finding_resolved=finding_passes,
        source_path=instance["source_path"],
        resource_budget=instance["resource_budget"],
        resolved_config=instance["environment"]["resolved_config"],
        clock_domain="monotonic",
    )
    result["test_kind"] = instance.get("test_kind", "functional")
    if "implementation" in instance:
        result["implementation"] = copy.deepcopy(instance["implementation"])
    return RunFinalizer().run(ctx, result, started, _policy)
