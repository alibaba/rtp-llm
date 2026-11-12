import asyncio
import time
from pathlib import Path

from ..config.validation import duration
from .environment import RunEnvironment
from .step.step import GateFailure, StepContext
from .step_trace import StepTrace, timestamp


class _PhaseTasks:
    def __init__(self):
        self._background = {}

    def start(self, step_id, operation):
        self._background[step_id] = asyncio.create_task(operation, name=step_id)

    def check(self):
        for task in self._background.values():
            if task.done():
                task.result()

    async def run(self, operation):
        pending = {task for task in self._background.values() if not task.done()}
        if not pending:
            return await operation
        foreground = asyncio.create_task(operation)
        try:
            while not foreground.done() and pending:
                _, pending = await asyncio.wait(
                    pending | {foreground}, return_when=asyncio.FIRST_COMPLETED
                )
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
            _, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            self.check()
        self.check()

    async def close(self):
        for task in self._background.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._background.values(), return_exceptions=True)


async def run_case(case, launch, directory):
    records = StepTrace(directory, case, launch)
    runtime = RunEnvironment(case, launch, records)
    results = {}
    result = {"status": "ERROR", "started_at": timestamp()}
    cancelled = False
    try:
        records.record_git_state(
            [
                Path(__file__).resolve().parents[3],
                case.directory,
                *[r["cwd"] for r in launch["components"].values()],
            ]
        )
        records.emit(event="setup_started")
        await runtime.start()
        records.emit(event="setup_finished")
        for phase in case.phases:
            phase_deadline = time.monotonic() + float(duration(phase.timeout))
            records.emit(
                event="phase_started", phase_id=phase.id, describe=phase.describe
            )
            phase_status = "ERROR"
            tasks = _PhaseTasks()
            try:
                for step in phase.steps:
                    tasks.check()
                    deadline = (
                        min(
                            phase_deadline,
                            time.monotonic() + float(duration(step.within)),
                        )
                        if step.within
                        else phase_deadline
                    )
                    context = StepContext(
                        runtime,
                        records,
                        timestamp(),
                        deadline,
                        phase_id=phase.id,
                        results=results,
                    )
                    if step.background:
                        tasks.start(step.id, step.run(context))
                    else:
                        await tasks.run(step.run(context))
                await tasks.finish()
                phase_status = "PASS"
            except GateFailure:
                phase_status = "FAIL"
                raise
            except asyncio.CancelledError:
                phase_status = "CANCELLED"
                raise
            finally:
                await tasks.close()
                records.emit(
                    event="phase_finished", phase_id=phase.id, status=phase_status
                )
        result["status"] = "PASS"
    except GateFailure as error:
        result.update(status="FAIL", error=str(error))
    except asyncio.CancelledError:
        cancelled = True
        result.update(status="CANCELLED", error="Execution cancelled")
    except Exception as error:
        result.update(status="ERROR", error=f"{type(error).__name__}: {error}")
    finally:
        cleanup = []
        try:
            records.emit(event="teardown_started")
        except OSError as error:
            cleanup.append(f"record teardown_started: {error}")
        closing = asyncio.create_task(runtime.close())
        while True:
            try:
                cleanup.extend(await asyncio.shield(closing))
                break
            except asyncio.CancelledError:
                cancelled = True
                result.update(
                    status="CANCELLED", error="Execution cancelled during teardown"
                )
        try:
            records.emit(event="teardown_finished", errors=cleanup)
        except OSError as error:
            cleanup.append(f"record teardown_finished: {error}")
        result["cleanup_errors"] = cleanup
        if cleanup:
            result["status"] = "ERROR"
        result["finished_at"] = timestamp()
        records.close(result)
    if cancelled:
        raise asyncio.CancelledError
    return result
