import asyncio
from pathlib import Path
import time

from ..config.validation import duration
from . import ExecutionError
from .environment import RunEnvironment
from .run_trace import RunTrace, timestamp, write_json
from .step.step import GateFailure, StepContext


class _PhaseTasks:
    def __init__(self):
        self._background = {}
        self._stopping = set()

    def start(self, step_id, operation):
        self._background[step_id] = asyncio.create_task(operation, name=step_id)

    async def stop(self, step_id):
        task = self._background.get(step_id)
        if task is None:
            raise ExecutionError(f'No background Step {step_id!r} in this Phase')
        if task.done():
            self._check(step_id, task)
            return False
        self._stopping.add(step_id)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._check(step_id, task)
        return task.cancelled()

    def _check(self, step_id, task):
        if task.done() and not (task.cancelled() and step_id in self._stopping):
            task.result()

    def check(self):
        for step_id, task in self._background.items():
            self._check(step_id, task)

    async def run(self, operation):
        pending = {task for task in self._background.values() if not task.done()}
        if not pending:
            return await operation
        foreground = asyncio.create_task(operation)
        try:
            while not foreground.done() and pending:
                _, pending = await asyncio.wait(pending | {foreground}, return_when=asyncio.FIRST_COMPLETED)
                self.check()
                pending.discard(foreground)
            return await foreground
        finally:
            if not foreground.done():
                foreground.cancel()
            await asyncio.gather(foreground, return_exceptions=True)

    async def finish(self):
        pending = {task for task in self._background.values() if not task.done()}
        while pending:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            self.check()
        self.check()

    async def close(self):
        for task in self._background.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background.values(), return_exceptions=True)


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
            tasks = _PhaseTasks()
            try:
                for step in phase.steps:
                    tasks.check()
                    deadline = min(phase_deadline, time.monotonic() + float(duration(step.within))) if step.within else phase_deadline
                    context = StepContext(runtime, records, timestamp(), deadline, tasks.stop)
                    if step.background:
                        tasks.start(step.id, step.run(context, results, phase.id))
                    else:
                        await tasks.run(step.run(context, results, phase.id))
                await tasks.finish()
                phase_status = 'PASS'
            except GateFailure:
                phase_status = 'FAIL'
                raise
            except asyncio.CancelledError:
                phase_status = 'CANCELLED'
                raise
            finally:
                await tasks.close()
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
