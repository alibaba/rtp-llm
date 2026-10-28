import asyncio
from datetime import datetime
import time
from pathlib import Path

from ..config.compiler import _value
from ..config.model import CallStep, CheckStep, Reference
from ..config.values import duration
from .assertions import compare
from .processes import Processes
from .records import Records, timestamp, write_json
from .transport import ExecutionError


def resolve(value, results):
    if isinstance(value, Reference):
        try:
            result = results[value.step]
            for part in value.path:
                result = result[part]
            return result
        except (KeyError, TypeError) as error:
            raise ExecutionError(f'Cannot resolve {value.step}.{value.path}') from error
    if isinstance(value, dict):
        return {key: resolve(item, results) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve(item, results) for item in value]
    return value


def _epoch(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ExecutionError('Component timestamps must include a timezone')
    return parsed.timestamp()


def observation(args, query_time, traffic_start):
    if 'range' not in args:
        return None
    selected = args['range']
    end = _epoch(query_time)
    if selected == 'latest':
        return {'selector': 'latest', 'at': query_time}
    if selected == 'all':
        if traffic_start is None:
            raise ExecutionError('range: all requires an actual traffic start receipt')
        start, stop = _epoch(traffic_start), end
    else:
        start, stop = (end + float(duration(item, negative=True)) for item in selected)
    if start > stop:
        raise ExecutionError('Observation range starts after its end')
    return {'selector': selected, 'start': start, 'end': stop}


async def run_case(case, registry, launch, directory):
    # Resolve every binding before starting a process or creating an experiment.
    for phase in case.phases:
        for step in phase.steps:
            if step.interface != 'runtime/wait' and not any(step.interface.startswith(p) for p in launch['bindings']):
                raise ExecutionError(f'No process binding for {step.interface}')
    records = Records(directory, case, launch)
    processes = Processes(case, launch, records)
    results, traffic_start = {}, None
    result = {'status': 'ERROR', 'started_at': timestamp()}
    cancelled = False
    try:
        records.versions([Path(__file__).resolve().parents[3], case.directory,
                          *[r['cwd'] for r in launch['components'].values()]])
        write_json(records.directory / 'interfaces.json', {'interfaces': registry.interfaces})
        records.emit(event='setup_started')
        await processes.start()
        records.emit(event='setup_finished')
        for phase in case.phases:
            phase_deadline = time.monotonic() + float(duration(phase.timeout))
            records.emit(event='phase_started', phase=phase.id, describe=phase.describe)
            phase_status = 'ERROR'
            try:
                for step in phase.steps:
                    args = resolve(step.args, results)
                    spec = registry.get(step.interface)
                    # Revalidate references using the same generated schema before any side effect.
                    args = _value(args, spec['args'], {}, step.id + '.args')
                    query_time = timestamp()
                    observed = observation(args, query_time, traffic_start)
                    budget = phase_deadline - time.monotonic()
                    if isinstance(step, CallStep) and step.within:
                        budget = min(budget, float(duration(step.within)))
                    records.emit(event='step_started', phase=phase.id, step=step.id, describe=step.describe,
                                 interface=step.interface, args=args, query_time=query_time, observation=observed)
                    gate = None
                    try:
                        if budget <= 0:
                            raise TimeoutError('Phase deadline exceeded')
                        if isinstance(step, CheckStep):
                            gate = {'step': step.id, 'interface': step.interface, 'args': args,
                                    'query_time': query_time, 'observation': observed,
                                    'expect': resolve(step.expect.comparisons, results)}
                        if step.interface == 'runtime/wait':
                            await processes.guard(asyncio.sleep(float(duration(args['duration']))), budget)
                            actual = {}
                        else:
                            payload = {'interface': step.interface, 'args': args,
                                       'context': {'run': step.run if isinstance(step, CallStep) else 'foreground',
                                                   'query_time': query_time, 'observation': observed,
                                                   'timeout_ms': max(1, int(budget * 1000))}}
                            actual = await processes.guard(processes.invoke(payload), budget)
                        actual = _value(actual, spec['returns'], {}, step.id + '.result')
                        if spec['returns'].get('name') == 'QueryResult' and _epoch(actual['query_time']) != _epoch(query_time):
                            raise ExecutionError('Query response does not match the fixed invocation time')
                        results[step.id] = actual
                        if step.interface == 'client/traffic/run':
                            _epoch(actual['started_at'])
                            traffic_start = actual['started_at']
                        if gate is not None:
                            gate['actual'] = actual
                            gate['status'] = 'PASS' if compare(actual, gate['expect'], spec) else 'FAIL'
                            records.emit('gates', **gate)
                            if gate['status'] == 'FAIL':
                                raise GateFailure(f'Gate {step.id} failed')
                        records.emit(event='step_finished', step=step.id, status='PASS', result=actual)
                    except BaseException as error:
                        failure_status = ('CANCELLED' if isinstance(error, asyncio.CancelledError) else
                                          'FAIL' if isinstance(error, GateFailure) else 'ERROR')
                        if gate is not None and 'status' not in gate:
                            gate.update(status=failure_status, error=str(error) or type(error).__name__)
                            records.emit('gates', **gate)
                        records.emit(event='step_finished', step=step.id,
                                     status=failure_status,
                                     error=str(error) or type(error).__name__,
                                     outcome_unknown=isinstance(step, CallStep) and step.interface != 'runtime/wait')
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
        closing = asyncio.create_task(processes.close())
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
