import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest import IsolatedAsyncioTestCase, main

from rtp_llm.multimodal.source_download_executor import SourceDownloadExecutor


class SourceDownloadExecutorTest(IsolatedAsyncioTestCase):
    async def test_zero_preserves_default_executor_and_context(self):
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(1, "test-shared"))
        pool = SourceDownloadExecutor()
        request_id = contextvars.ContextVar("request_id", default=None)
        request_id.set("request-1")
        try:
            name, value = await pool.run(
                lambda: (threading.current_thread().name, request_id.get())
            )
            self.assertTrue(name.startswith("test-shared"))
            self.assertEqual(value, "request-1")
        finally:
            pool.shutdown()
        self.assertEqual(await asyncio.to_thread(lambda: 7), 7)

    async def test_download_limit_does_not_block_default_executor(self):
        pool = SourceDownloadExecutor(2)
        release, entered = threading.Event(), threading.Event()
        lock = threading.Lock()
        active, peak, started = 0, 0, 0
        request_id = contextvars.ContextVar("request_id", default=None)
        request_id.set("request-2")

        def download(*, suffix):
            nonlocal active, peak, started
            with lock:
                active += 1
                started += 1
                peak = max(peak, active)
                if active == 2:
                    entered.set()
            try:
                if not release.wait(5):
                    raise TimeoutError("test downloads were not released")
                return request_id.get() + suffix
            finally:
                with lock:
                    active -= 1

        tasks = [
            asyncio.create_task(pool.run(download, suffix="-done")) for _ in range(3)
        ]
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            self.assertEqual(started, 2)
            self.assertEqual(
                await asyncio.wait_for(asyncio.to_thread(lambda: "upload"), 1),
                "upload",
            )
            release.set()
            self.assertEqual(await asyncio.gather(*tasks), ["request-2-done"] * 3)
            self.assertEqual(peak, 2)
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.shutdown()

    async def test_cancelled_queued_download_never_runs(self):
        pool = SourceDownloadExecutor(1)
        release, entered, queued_ran = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def download():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test download was not released")

        active = asyncio.create_task(pool.run(download))
        queued = None
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            queued = asyncio.create_task(pool.run(queued_ran.set))
            await asyncio.sleep(0)
            queued.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await queued
            release.set()
            await active
            pool.shutdown()
            self.assertFalse(queued_ran.is_set())
        finally:
            release.set()
            await asyncio.gather(
                active, *([queued] if queued else []), return_exceptions=True
            )
            pool.shutdown()

    async def test_shutdown_drains_running_download_and_rejects_new_work(self):
        pool = SourceDownloadExecutor(1)
        release, entered, drained = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def download():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test download was not released")
            drained.set()

        task = asyncio.create_task(pool.run(download))
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            release.set()
            await asyncio.to_thread(pool.shutdown)
            self.assertTrue(drained.is_set())
            await task
            with self.assertRaisesRegex(RuntimeError, "closed"):
                await pool.run(lambda: None)
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            pool.shutdown()

    async def test_download_failure_propagates_and_worker_remains_usable(self):
        pool = SourceDownloadExecutor(1)

        def fail():
            raise TimeoutError("source timeout")

        try:
            with self.assertRaisesRegex(TimeoutError, "source timeout"):
                await pool.run(fail)
            self.assertEqual(await pool.run(lambda: 42), 42)
        finally:
            pool.shutdown()

    async def test_negative_concurrency_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SourceDownloadExecutor(-1)


if __name__ == "__main__":
    main()
