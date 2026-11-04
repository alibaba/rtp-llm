import asyncio
from pathlib import Path
import time

from ..config.validation import duration
from .environment import RunEnvironment
from .run_trace import RunTrace, timestamp, write_json
from .step.step import GateFailure, StepContext


async def run_case(case, registry, launch, directory):
    records = RunTrace(directory, case, launch)
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
                    deadline = min(phase_deadline, time.monotonic() + float(duration(step.within))) if step.within else phase_deadline
                    context = StepContext(runtime, records, timestamp(), deadline)
                    await step.run(context, results, phase.id)
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
