"""Bound Case steps and the common execution lifecycle."""
from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import time
from typing import Any, ClassVar

from ...config.validation import duration, validate
from .. import ExecutionError
from .assertions import matches_expectation


def resolve_refs(value, results, case_globals):
    if isinstance(value, Reference):
        try:
            result = case_globals if value.source == 'globals' else results[value.source]
            for part in value.path:
                result = result[part]
            return result
        except (KeyError, TypeError) as error:
            raise ExecutionError(f'Cannot resolve {value.source}.{value.path}') from error
    if isinstance(value, dict):
        return {k: resolve_refs(v, results, case_globals) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, results, case_globals) for v in value]
    return value


@dataclass(frozen=True)
class Reference:
    source: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class Expect:
    comparisons: dict[str, Any]


@dataclass(frozen=True)
class Step(ABC):
    definition: ClassVar
    id: str
    describe: str
    interface: str
    args: dict[str, Any]
    expect: Expect | None = None
    within: str | None = None
    wait_after: str = '0s'
    background: bool = False

    @staticmethod
    @abstractmethod
    async def execute(context, arguments):
        """Implement the action for one set of validated arguments."""

    async def run(self, context):
        with context.records.step(context.phase_id, self.id, self.interface):
            await self._run(context)

    async def _run(self, context):
        results, phase_id = context.results, context.phase_id
        started = time.monotonic()
        interface = self.definition
        runtime, records = context.runtime, context.records
        gate, invoked, returned = None, False, False
        stage = 'arguments'
        records.emit(event='step_started', phase=phase_id, step=self.id, interface=self.interface,
                     describe=self.describe, query_time=context.query_time)
        try:
            args = resolve_refs(self.args, results, context.globals())
            arguments = interface.arguments(args)
            expected = resolve_refs(self.expect.comparisons, results, context.globals()) if self.expect else None
            if expected is not None:
                for op, value in expected.items():
                    expected_type = interface.returns
                    if op == 'count':
                        expected_type = {'type': 'integer', 'minimum': 0}
                    elif op in ('gte', 'lte'):
                        expected_type = {'type': 'number'}
                    validate(value, expected_type, 'expect.' + op)
                gate = {'step': self.id, 'interface': self.interface, 'args': args,
                        'query_time': context.query_time, 'expect': expected}
            budget = context.timeout
            stage = 'invoke'
            invoked = True
            actual = await runtime.supervisor.guard(interface.call(context, arguments), budget)
            returned = True
            stage = 'expect'
            if gate is not None:
                gate.update(actual=actual, status='PASS' if matches_expectation(actual, expected) else 'FAIL')
                records.emit('gates', **gate)
                if gate['status'] == 'FAIL':
                    raise GateFailure(f'Gate {self.id} failed')
            wait = float(duration(self.wait_after))
            if wait > 0:
                stage = 'wait_after'
                records.emit(event='wait_started', step=self.id, duration=self.wait_after)
                budget = context.timeout
                await runtime.supervisor.guard(asyncio.sleep(wait), budget)
            results[self.id] = actual
            records.emit(event='step_finished', step=self.id, status='PASS', args=args, result=actual,
                         duration_ms=round((time.monotonic() - started) * 1000, 3))
        except BaseException as error:
            status = ('CANCELLED' if isinstance(error, asyncio.CancelledError) else
                      'FAIL' if isinstance(error, GateFailure) else 'ERROR')
            if gate is not None and 'status' not in gate:
                gate.update(status=status, error=str(error) or type(error).__name__)
                records.emit('gates', **gate)
            records.emit(event='step_finished', step=self.id, status=status,
                         error=str(error) or type(error).__name__,
                         error_stage=stage, duration_ms=round((time.monotonic() - started) * 1000, 3),
                         outcome_unknown=invoked and not returned and interface.effect == 'write',
                         **({'result': actual} if returned else {}))
            raise


@dataclass(frozen=True)
class StepContext:
    runtime: Any
    records: Any
    query_time: str
    deadline: float
    phase_id: str = ''
    results: dict[str, Any] = field(default_factory=dict)

    @property
    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Step deadline exceeded')
        return remaining

    def globals(self):
        return {name: sorted(item['rpc'] for item in self.runtime.topology.active() if item['role'] == role)
                for name, role in [('masters', 'master'), ('prefill', 'prefill'), ('decode', 'decode'), ('vit', 'vit')]}


class GateFailure(ExecutionError):
    pass
