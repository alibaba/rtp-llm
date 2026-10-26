"""Second, small completion channel for inspection in CPU preprocess workers.

The Pool still transports model CPU tensors. Encoded source bytes never enter
this channel: only task IDs and verdicts do. Shared generation-tagged slots
provide bounded ownership and cancellation without a manager process.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing.util
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from rtp_llm.multimodal.greennet_hook import GreenNetVerdict


@dataclass(frozen=True)
class WorkerInspectionTask:
    slot: int
    task_id: int
    deadline: float
    request: Any


class WorkerInspectionChannel:
    """Parent owns Futures; children inherit only queue and cancellation slots."""

    def __init__(self, context, factory, capacity, live_pids, worker_exited):
        self.factory = factory
        self.results = context.Queue()
        self.tokens = context.Array("q", max(1, capacity))
        # Slot ownership, not a semaphore: a killed worker must not leak a
        # permit forever. All accesses use the cancellation array's shared lock.
        self.running = context.RawArray("q", len(self.tokens))
        self.owners = context.RawArray("i", len(self.tokens))
        self.max_concurrency = max(
            1, min(capacity, getattr(factory, "max_concurrency", capacity))
        )
        self._free = list(range(len(self.tokens)))
        self._pending = {}
        self._next_id = 0
        self._condition = threading.Condition()
        self._closed = False
        self._live_pids = live_pids
        self._worker_exited = worker_exited
        self._thread = threading.Thread(
            target=self._collect, name="mm-inspection-results", daemon=True
        )
        self._thread.start()

    @property
    def worker_args(self):
        return (
            self.factory,
            self.results,
            self.tokens,
            self.running,
            self.owners,
            self.max_concurrency,
        )

    def register(self, request, deadline):
        with self._condition:
            while not self._free and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise concurrent.futures.TimeoutError(
                        "CPU inspection admission deadline exceeded"
                    )
                self._condition.wait(remaining)
            if self._closed:
                raise RuntimeError("CPU inspection channel is closed")
            if deadline <= time.monotonic():
                raise concurrent.futures.TimeoutError(
                    "CPU inspection deadline exceeded"
                )
            slot = self._free.pop()
            self._next_id += 1
            task = WorkerInspectionTask(slot, self._next_id, deadline, request)
            future = concurrent.futures.Future()
            self.tokens[slot] = task.task_id
            self._pending[task.task_id] = (task, future)

        def cancelled(completed):
            if completed.cancelled():
                with self._condition:
                    if task.task_id in self._pending:
                        self.tokens[slot] = -task.task_id

        future.add_done_callback(cancelled)
        return task, future

    def finish(self, task_id, verdict=None, error=None):
        with self._condition:
            pending = self._pending.pop(task_id, None)
            if pending is None:
                return
            task, future = pending
            with self.tokens.get_lock():
                self.tokens[task.slot] = 0
                self.running[task.slot] = 0
                self.owners[task.slot] = 0
            self._free.append(task.slot)
            self._condition.notify_all()
        # Future callbacks may re-enter the channel or the engine. No locks held.
        try:
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(verdict)
        except concurrent.futures.InvalidStateError:
            pass  # Request cancellation won the race.

    def _collect(self):
        while True:
            with self._condition:
                if self._closed:
                    return
            try:
                kind, task_id, value = self.results.get(timeout=0.1)
                if kind == "done":
                    if isinstance(value, GreenNetVerdict):
                        self.finish(task_id, verdict=value)
                    else:
                        kind, message = value
                        error = (
                            concurrent.futures.TimeoutError(message)
                            if kind == "timeout"
                            else RuntimeError(message)
                        )
                        self.finish(task_id, error=error)
            except queue.Empty:
                pass
            except (EOFError, OSError):
                self.fail_all(RuntimeError("CPU inspection result channel failed"))
                return
            now = time.monotonic()
            alive = set(self._live_pids())
            with self._condition:
                pending = list(self._pending.items())
            for task_id, (task, _) in pending:
                with self._condition:
                    if task_id not in self._pending:
                        continue
                    pid = self.owners[task.slot]
                if now >= task.deadline:
                    self.finish(
                        task_id,
                        error=concurrent.futures.TimeoutError(
                            "CPU inspection timed out"
                        ),
                    )
                elif pid and pid not in alive:
                    self._worker_exited()
                    self.finish(
                        task_id, error=RuntimeError("CPU inspection worker exited")
                    )

    def fail_all(self, error):
        with self._condition:
            task_ids = list(self._pending)
        for task_id in task_ids:
            self.finish(task_id, error=error)

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.fail_all(RuntimeError("CPU inspection channel closed"))
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2)
        self.results.close()
        self.results.join_thread()


class WorkerInspectionRunner:
    """One persistent event loop/provider per CPU worker, created on first use."""

    def __init__(self, factory, results, tokens, running, owners, max_concurrency):
        self.factory, self.results, self.tokens = factory, results, tokens
        self.running, self.max_concurrency = running, max_concurrency
        self.owners = owners
        self.loop = None
        self.thread = None
        self.provider = None
        multiprocessing.util.Finalize(self, self.close, exitpriority=10)

    def active(self, task):
        return (
            self.tokens[task.slot] == task.task_id and time.monotonic() < task.deadline
        )

    def started(self, task):
        # Do not rely on Queue's feeder flushing a "started" message before a
        # process crash; the parent must still know whose verdict was lost.
        with self.tokens.get_lock():
            if not self.active(task):
                raise TimeoutError("CPU inspection task cancelled or expired")
            self.owners[task.slot] = os.getpid()

    def _acquire(self, task, image_count):
        with self.tokens.get_lock():
            if not self.active(task):
                return False
            # A multi-image task consumes image credits, not just one job.
            # A larger group runs alone, with the provider's connector bound.
            count = min(image_count, self.max_concurrency)
            if sum(self.running) + count > self.max_concurrency:
                return False
            self.running[task.slot] = count
            return True

    def failed(self, task, error):
        # Never pickle arbitrary exceptions (which can own images or clients).
        kind = "timeout" if time.monotonic() >= task.deadline else "error"
        self.results.put(
            ("done", task.task_id, (kind, f"CPU inspection failed: {error}"))
        )

    def submit(self, task, mm_inputs, sources):
        if self.loop is None:
            self.loop = asyncio.new_event_loop()

            def run():
                asyncio.set_event_loop(self.loop)
                self.loop.run_forever()

            self.thread = threading.Thread(
                target=run, name="mm-worker-inspection", daemon=True
            )
            self.thread.start()
        asyncio.run_coroutine_threadsafe(
            self._inspect(task, mm_inputs, sources), self.loop
        )

    async def _inspect(self, task, mm_inputs, sources):
        handle = None

        async def inspect():
            nonlocal handle
            if self.provider is None:
                self.provider = self.factory()
            handle = await self.provider.submit_sources(
                task.request, mm_inputs, sources
            )
            sources.clear()
            verdict = await handle.wait_result()
            if not isinstance(verdict, GreenNetVerdict):
                raise RuntimeError("CPU inspection returned no verdict")
            return verdict

        running = None
        acquired = False
        try:
            while self.active(task):
                if self._acquire(task, len(mm_inputs)):
                    acquired = True
                    break
                await asyncio.sleep(0.01)
            if not self.active(task):
                raise asyncio.CancelledError()
            running = asyncio.create_task(inspect())
            while self.active(task):
                done, _ = await asyncio.wait([running], timeout=0.05)
                if done:
                    self.results.put(("done", task.task_id, running.result()))
                    return
            raise asyncio.CancelledError()
        except asyncio.CancelledError:
            self.failed(task, "cancelled or expired")
        except Exception as error:
            self.failed(task, error)
        finally:
            if running is not None and not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)
            if handle is not None:
                handle.cancel()
            sources.clear()
            if acquired:
                with self.tokens.get_lock():
                    if abs(self.tokens[task.slot]) == task.task_id:
                        self.running[task.slot] = 0

    def close(self):
        if self.loop is None:
            return

        async def drain():
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.provider is not None:
                await self.provider.close()
            await self.loop.shutdown_asyncgens()

        try:
            asyncio.run_coroutine_threadsafe(drain(), self.loop).result(timeout=2)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=2)
            if not self.thread.is_alive():
                self.loop.close()
