"""Bound Case steps and the common execution lifecycle."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ...config.validation import duration, expectation_operand_schema, validate
from .. import ExecutionError
from .step_registry import StepDefinition, to_record_value


def resolve_refs(value, results, case_globals):
    if isinstance(value, Reference):
        try:
            result = (
                case_globals if value.source == "globals" else results[value.source]
            )
            for part in value.path:
                result = result[part]
            return result
        except (KeyError, TypeError) as error:
            raise ExecutionError(
                f"Cannot resolve {value.source}.{value.path}"
            ) from error
    if isinstance(value, dict):
        return {k: resolve_refs(v, results, case_globals) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, results, case_globals) for v in value]
    return value


def matches_expectation(actual, expected):
    for operator, target in expected.items():
        if operator == "eq":
            good = actual == target
        elif operator == "count":
            good = len(actual) == target
        elif operator == "gte":
            values = actual if isinstance(actual, list) else [actual]
            good = bool(values) and all(value >= target for value in values)
        else:
            values = actual if isinstance(actual, list) else [actual]
            good = bool(values) and all(value <= target for value in values)
        if not good:
            return False
    return True


@dataclass(frozen=True)
class Reference:
    source: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class Expect:
    comparisons: dict[str, Any]


@dataclass(frozen=True)
class Step:
    definition: StepDefinition = field(kw_only=True, repr=False)
    id: str
    describe: str
    args: dict[str, Any]
    expect: Expect | None = None
    within: str | None = None
    wait_after: str = "0s"
    background: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "describe": self.describe,
            "interface": self.definition.name,
            "args": to_record_value(self.args),
            "expect": to_record_value(self.expect),
            "within": self.within,
            "wait_after": self.wait_after,
            "background": self.background,
        }

    async def run(self, context):
        with context.records.step(context.phase_id, self.id, self.definition.name):
            await self._run(context)

    async def _run(self, context):
        results = context.results
        started = time.monotonic()
        definition = self.definition
        runtime, records = context.runtime, context.records
        gate, invoked, returned = None, False, False
        stage = "arguments"
        records.emit(event="step_started")
        try:
            args = resolve_refs(self.args, results, context.globals())
            arguments = definition.arguments(args)
            expected = (
                resolve_refs(self.expect.comparisons, results, context.globals())
                if self.expect
                else None
            )
            if expected is not None:
                for op, value in expected.items():
                    validate(
                        value,
                        expectation_operand_schema(op, definition.returns),
                        "expect." + op,
                    )
                gate = {
                    "unit": definition.unit,
                    "args": args,
                    "query_time": context.query_time,
                    "expect": expected,
                }
            budget = context.timeout
            stage = "invoke"
            invoked = True
            actual = await runtime.supervisor.guard(
                definition.call(context, arguments), budget
            )
            returned = True
            stage = "expect"
            if gate is not None:
                gate.update(
                    actual=actual,
                    status="PASS" if matches_expectation(actual, expected) else "FAIL",
                )
                records.emit("gates", **gate)
                if gate["status"] == "FAIL":
                    raise GateFailure(f"Gate {self.id} failed")
            wait = float(duration(self.wait_after))
            if wait > 0:
                stage = "wait_after"
                records.emit(event="wait_started", duration=self.wait_after)
                budget = context.timeout
                await runtime.supervisor.guard(asyncio.sleep(wait), budget)
            results[self.id] = actual
            records.emit(
                event="step_finished",
                status="PASS",
                args=args,
                result=actual,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )
        except BaseException as error:
            status = (
                "CANCELLED"
                if isinstance(error, asyncio.CancelledError)
                else "FAIL" if isinstance(error, GateFailure) else "ERROR"
            )
            if gate is not None and "status" not in gate:
                gate.update(status=status, error=str(error) or type(error).__name__)
                records.emit("gates", **gate)
            records.emit(
                event="step_finished",
                status=status,
                error=str(error) or type(error).__name__,
                error_stage=stage,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                outcome_unknown=invoked
                and not returned
                and definition.effect == "write",
                **({"result": actual} if returned else {}),
            )
            raise


@dataclass(frozen=True)
class StepContext:
    runtime: Any
    records: Any
    query_time: str
    deadline: float
    phase_id: str = ""
    results: dict[str, Any] = field(default_factory=dict)

    @property
    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Step deadline exceeded")
        return remaining

    def globals(self):
        nodes = self.runtime.topology.active()
        return {
            name: sorted(item["rpc"] for item in nodes if item["role"] == role)
            for name, role in [
                ("masters", "master"),
                ("prefill", "prefill"),
                ("decode", "decode"),
                ("vit", "vit"),
            ]
        }


class GateFailure(ExecutionError):
    pass
