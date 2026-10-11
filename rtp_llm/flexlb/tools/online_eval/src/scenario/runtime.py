"""Run immutable compiled instances through owned sessions and finalization."""

from __future__ import annotations

import time
from runtime.deadline import Deadline, StageTimeout, interruptible
from scenario.context import RuntimeContext
from scenario.stage_execution import StageExecutor
from scenario.finalization import RunFinalizer
from scenario.plan import CompiledInstance
from scenario.run_result import RunResult


class ScenarioRunner:
    def run(self, instance, backend, handlers=None, artifact_dir=".", clock=time.monotonic,
            sleeper=time.sleep, cancelled=None, enforce_deadlines=False, policy=None) -> RunResult:
        plan = CompiledInstance.read(instance)
        ctx = RuntimeContext(plan.to_dict(), backend, artifact_dir, clock, sleeper, enforce_deadlines)
        ctx.plan = plan
        ctx.artifact_dir.mkdir(parents=True, exist_ok=True)
        started = clock()
        end = started + plan.execution.timeout_s
        ctx.instance_deadline_s = end
        execution = StageExecutor().run(ctx, dict(handlers or {}), started, end, cancelled, policy)
        result = RunResult(plan, execution, int((clock()-started)*1000), execution.error)
        return RunFinalizer().run(ctx, result, started, policy)


def execute_instance(instance, backend, handlers=None, artifact_dir=".", clock=time.monotonic,
                     sleeper=time.sleep, cancelled=None, enforce_deadlines=False, _policy=None):
    """Artifact-facing API; execution itself operates on CompiledInstance and RunResult."""
    return ScenarioRunner().run(instance, backend, handlers, artifact_dir, clock, sleeper,
                                cancelled, enforce_deadlines, _policy).to_dict()
