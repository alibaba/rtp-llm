import asyncio
from dataclasses import dataclass
import time
from pathlib import Path

from ..config.model import Reference
from ..config.validation import duration, validate
from .assertions import compare
from .environment import RunEnvironment
from .records import Records, timestamp, write_json
from .transport import ExecutionError


def resolve(value, results, globals):
    if isinstance(value, Reference):
        try:
            result = globals if value.source == 'globals' else results[value.source]
            for part in value.path:
                result = result[part]
            return result
        except (KeyError, TypeError) as error:
            raise ExecutionError(f'Cannot resolve {value.source}.{value.path}') from error
    if isinstance(value, dict):
        return {k: resolve(v, results, globals) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results, globals) for v in value]
    return value


@dataclass(frozen=True)
class CaseContext:
    runtime: RunEnvironment
    records: Records
    query_time: str
    deadline: float

    @property
    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Step deadline exceeded')
        return remaining

    def globals(self):
        return {name: sorted(item['rpc'] for item in self.runtime.topology.active() if item['role'] == role)
                for name, role in [('masters', 'master'), ('prefill', 'prefill'), ('decode', 'decode'), ('vit', 'vit')]}


async def run_case(case, registry, launch, directory):
    records = Records(directory, case, launch)
    runtime = RunEnvironment(case, launch, records)
    results = {}
    result = {'status': 'ERROR', 'started_at': timestamp()}
    cancelled = False
    try:
        records.versions([Path(__file__).resolve().parents[3], case.directory,
                          *[r['cwd'] for r in launch['components'].values()]])
        write_json(records.directory / 'interfaces.json', {'interfaces': registry.interfaces})
        records.emit(event='setup_started')
        await runtime.start()
        records.emit(event='setup_finished')
        for phase in case.phases:
            phase_deadline = time.monotonic() + float(duration(phase.timeout))
            records.emit(event='phase_started', phase=phase.id, describe=phase.describe)
            phase_status = 'ERROR'
            try:
                for step in phase.steps:
                    interface = registry.get(step.interface)
                    deadline = min(phase_deadline, time.monotonic() + float(duration(step.within))) if step.within else phase_deadline
                    context = CaseContext(runtime, records, timestamp(), deadline)
                    gate, invoked, returned = None, False, False
                    stage = 'arguments'
                    records.emit(event='step_started', phase=phase.id, step=step.id, interface=step.interface,
                                 describe=step.describe, query_time=context.query_time)
                    try:
                        args = resolve(step.args, results, context.globals())
                        arguments = interface.arguments(args)
                        expected = resolve(step.expect.comparisons, results, context.globals()) if step.expect else None
                        if expected is not None:
                            for op, value in expected.items():
                                validate(value, {'type': 'integer', 'minimum': 0} if op == 'count' else interface.returns, 'expect.' + op)
                            gate = {'step': step.id, 'interface': step.interface, 'args': args,
                                    'query_time': context.query_time, 'expect': expected}
                        budget = context.timeout
                        stage = 'invoke'
                        invoked = True
                        actual = await runtime.supervisor.guard(interface.call(context, arguments), budget)
                        returned = True
                        results[step.id] = actual
                        stage = 'expect'
                        if gate is not None:
                            gate.update(actual=actual, status='PASS' if compare(actual, expected) else 'FAIL')
                            records.emit('gates', **gate)
                            if gate['status'] == 'FAIL':
                                raise GateFailure(f'Gate {step.id} failed')
                        wait = float(duration(step.wait_after))
                        if wait > 0:
                            stage = 'wait_after'
                            records.emit(event='wait_started', step=step.id, duration=step.wait_after)
                            budget = context.timeout
                            await runtime.supervisor.guard(asyncio.sleep(wait), budget)
                        records.emit(event='step_finished', step=step.id, status='PASS', args=args, result=actual)
                    except BaseException as error:
                        status = ('CANCELLED' if isinstance(error, asyncio.CancelledError) else
                                  'FAIL' if isinstance(error, GateFailure) else 'ERROR')
                        if gate is not None and 'status' not in gate:
                            gate.update(status=status, error=str(error) or type(error).__name__)
                            records.emit('gates', **gate)
                        records.emit(event='step_finished', step=step.id, status=status,
                                     error=str(error) or type(error).__name__,
                                     error_stage=stage,
                                     outcome_unknown=invoked and not returned and interface.effect == 'write',
                                     **({'result': actual} if returned else {}))
                        raise
                phase_status = 'PASS'
            except GateFailure:
                phase_status = 'FAIL'
                raise
            except asyncio.CancelledError:
                phase_status = 'CANCELLED'
                raise
            finally:
                records.emit(event='phase_finished', phase=phase.id, status=phase_status)
        result['status'] = 'PASS'
    except GateFailure as error:
        result.update(status='FAIL', error=str(error))
    except asyncio.CancelledError:
        cancelled = True
        result.update(status='CANCELLED', error='Execution cancelled')
    except Exception as error:
        result.update(status='ERROR', error=f'{type(error).__name__}: {error}')
    finally:
        records.emit(event='teardown_started')
        closing = asyncio.create_task(runtime.close())
        while True:
            try:
                cleanup = await asyncio.shield(closing)
                break
            except asyncio.CancelledError:
                cancelled = True
                result.update(status='CANCELLED', error='Execution cancelled during teardown')
        result['cleanup_errors'] = cleanup
        if cleanup:
            result['status'] = 'ERROR'
        records.emit(event='teardown_finished', errors=cleanup)
        result['finished_at'] = timestamp()
        records.close(result)
    if cancelled:
        raise asyncio.CancelledError
    return result


class GateFailure(ExecutionError):
    pass
