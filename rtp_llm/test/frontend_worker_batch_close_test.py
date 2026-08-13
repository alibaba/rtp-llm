import asyncio
from unittest import IsolatedAsyncioTestCase, main

from rtp_llm.frontend.frontend_worker import FrontendWorker, PipelineResponse


class _FailingIterator:
    def __init__(self, error):
        self.error = error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.error

    async def aclose(self):
        self.close_calls += 1


class _BlockingIterator:
    def __init__(self, close_error=None):
        self.next_started = asyncio.Event()
        self.next_cancelled = asyncio.Event()
        self.close_error = close_error
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.next_cancelled.set()
            raise
        raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class _FiniteIterator:
    def __init__(self, values):
        self.values = iter(values)
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.values)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.close_calls += 1


class FrontendWorkerBatchCloseTest(IsolatedAsyncioTestCase):
    def make_batch(self, generators):
        worker = object.__new__(FrontendWorker)
        return worker._parallel_batch_async_generators(
            incremental=False,
            generators=generators,
            batch_infer=True,
        )

    async def test_child_failure_cancels_and_closes_blocked_sibling(self):
        primary_error = RuntimeError("child failed")
        failed = _FailingIterator(primary_error)
        blocked = _BlockingIterator()
        batch = self.make_batch([failed, blocked])

        with self.assertRaises(RuntimeError) as raised:
            await asyncio.wait_for(batch.__anext__(), timeout=1)

        self.assertIs(raised.exception, primary_error)
        self.assertTrue(blocked.next_started.is_set())
        self.assertTrue(blocked.next_cancelled.is_set())
        self.assertEqual(failed.close_calls, 1)
        self.assertEqual(blocked.close_calls, 1)

    async def test_external_cancellation_closes_every_child(self):
        first = _BlockingIterator()
        second = _BlockingIterator()
        batch = self.make_batch([first, second])
        next_task = asyncio.create_task(batch.__anext__())
        await asyncio.gather(first.next_started.wait(), second.next_started.wait())

        next_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(next_task, timeout=1)

        self.assertTrue(first.next_cancelled.is_set())
        self.assertTrue(second.next_cancelled.is_set())
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(second.close_calls, 1)

    async def test_cleanup_failure_dominates_child_failure(self):
        primary_error = RuntimeError("child failed")
        close_error = RuntimeError("sibling close failed")
        failed = _FailingIterator(primary_error)
        blocked = _BlockingIterator(close_error=close_error)
        batch = self.make_batch([failed, blocked])

        with self.assertRaises(RuntimeError) as raised:
            await asyncio.wait_for(batch.__anext__(), timeout=1)

        self.assertIs(raised.exception, close_error)
        self.assertIs(raised.exception.__cause__, primary_error)
        self.assertTrue(blocked.next_cancelled.is_set())
        self.assertEqual(failed.close_calls, 1)
        self.assertEqual(blocked.close_calls, 1)

    async def test_value_and_stop_preserve_round_result(self):
        first_response = PipelineResponse(response="first")
        first = _FiniteIterator([first_response])
        second = _FiniteIterator([])
        batch = self.make_batch([first, second])

        result = await asyncio.wait_for(batch.__anext__(), timeout=1)
        self.assertEqual(result.response_batch[0].response, "first")
        self.assertIsNotNone(result.response_batch[1])

        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(batch.__anext__(), timeout=1)
        self.assertEqual(first.close_calls, 1)
        self.assertEqual(second.close_calls, 1)


if __name__ == "__main__":
    main()
