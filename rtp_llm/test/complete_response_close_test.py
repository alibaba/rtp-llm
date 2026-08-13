import asyncio
from unittest import IsolatedAsyncioTestCase, main

from rtp_llm.utils.complete_response_async_generator import (
    CloseDependencyRegistry,
    CompleteResponseAsyncGenerator,
    _CloseState,
    _ManagedCloseError,
)


async def collect_last(responses):
    response = None
    async for response in responses:
        pass
    return response


class CloseTrackedAsyncIterator:
    def __init__(self, values):
        self._values = iter(values)
        self._closed = False
        self._wait_forever = asyncio.Event()
        self.blocked = asyncio.Event()
        self.close_calls = 0
        self.release_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            return next(self._values)
        except StopIteration:
            self.blocked.set()
            await self._wait_forever.wait()
            raise AssertionError("unreachable")

    async def aclose(self):
        self.close_calls += 1
        if not self._closed:
            self._closed = True
            self.release_count += 1


class FailOnceCloseAsyncIterator(CloseTrackedAsyncIterator):
    async def aclose(self):
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("injected close failure")
        if not self._closed:
            self._closed = True
            self.release_count += 1


class ControlledCloseAsyncIterator(CloseTrackedAsyncIterator):
    def __init__(self, values):
        super().__init__(values)
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_cancel_count = 0

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        try:
            await self.allow_close.wait()
        except asyncio.CancelledError:
            self.close_cancel_count += 1
            raise
        if not self._closed:
            self._closed = True
            self.release_count += 1


class GatedNextAsyncIterator(CloseTrackedAsyncIterator):
    def __init__(self, value):
        super().__init__([])
        self.value = value
        self.next_started = asyncio.Event()
        self.allow_next = asyncio.Event()

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        self.next_started.set()
        await self.allow_next.wait()
        if self._closed:
            raise StopAsyncIteration
        return self.value


class CompleteResponseCloseTest(IsolatedAsyncioTestCase):
    async def test_natural_completion_releases_source_once(self):
        release_count = 0

        async def source():
            nonlocal release_count
            try:
                yield "first"
                yield "second"
            finally:
                release_count += 1

        response = CompleteResponseAsyncGenerator(source(), collect_last)
        self.assertEqual([item async for item in response], ["first", "second"])
        await response.aclose()
        await response.aclose()
        self.assertEqual(release_count, 1)

    async def test_concurrent_close_releases_each_source_once(self):
        sources = [CloseTrackedAsyncIterator([index]) for index in range(64)]
        responses = []
        for source in sources:
            response = CompleteResponseAsyncGenerator(source, collect_last)
            await response.__anext__()
            responses.append(response)

        await asyncio.gather(
            *(response.aclose() for response in responses for _ in range(2))
        )

        self.assertTrue(all(source.close_calls == 1 for source in sources))
        self.assertTrue(all(source.release_count == 1 for source in sources))

    async def test_failed_close_can_be_retried(self):
        source = FailOnceCloseAsyncIterator(["first"])
        response = CompleteResponseAsyncGenerator(source, collect_last)
        await response.__anext__()

        with self.assertRaisesRegex(RuntimeError, "injected close failure"):
            await response.aclose()
        await response.aclose()

        self.assertEqual(source.close_calls, 2)
        self.assertEqual(source.release_count, 1)

    async def test_primary_failure_still_closes_every_dependency(self):
        source = FailOnceCloseAsyncIterator(["first"])
        dependencies = [
            CloseTrackedAsyncIterator([]),
            CloseTrackedAsyncIterator([]),
        ]
        response = CompleteResponseAsyncGenerator(
            source,
            collect_last,
            close_dependencies=dependencies,
        )
        await response.__anext__()

        with self.assertRaisesRegex(RuntimeError, "injected close failure"):
            await response.aclose()

        self.assertTrue(all(item.close_calls == 1 for item in dependencies))
        self.assertTrue(all(item.release_count == 1 for item in dependencies))

    async def test_cancelled_waiter_does_not_cancel_shared_close(self):
        source = ControlledCloseAsyncIterator(["first"])
        response = CompleteResponseAsyncGenerator(source, collect_last)
        await response.__anext__()

        cancelled_waiter = asyncio.create_task(response.aclose())
        await source.close_started.wait()
        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter

        surviving_waiter = asyncio.create_task(response.aclose())
        source.allow_close.set()
        await surviving_waiter

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.close_cancel_count, 0)
        self.assertEqual(source.release_count, 1)

    async def test_cancelled_managed_waiter_does_not_mark_close_failed(self):
        source = ControlledCloseAsyncIterator([])
        registry = CloseDependencyRegistry()
        managed_source = registry.wrap(source)

        cancelled_waiter = asyncio.create_task(managed_source.aclose())
        await source.close_started.wait()
        cancelled_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled_waiter

        source.allow_close.set()
        await managed_source.aclose()
        await registry.aclose()

        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.close_cancel_count, 0)
        self.assertEqual(source.release_count, 1)
        self.assertTrue(managed_source._entry.closed)

    async def test_close_cancels_in_flight_next_before_release(self):
        source = CloseTrackedAsyncIterator(["first"])
        response = CompleteResponseAsyncGenerator(source, collect_last)
        await response.__anext__()

        pending_next = asyncio.create_task(response.__anext__())
        await source.blocked.wait()
        await response.aclose()

        with self.assertRaises(asyncio.CancelledError):
            await pending_next
        self.assertEqual(source.close_calls, 1)
        self.assertEqual(source.release_count, 1)

    async def test_next_and_close_registration_race_never_hangs(self):
        for index in range(100):
            source = GatedNextAsyncIterator(index)
            response = CompleteResponseAsyncGenerator(source, collect_last)
            pending_next = asyncio.create_task(response.__anext__())
            await source.next_started.wait()

            if index % 2 == 0:
                close_task = asyncio.create_task(response.aclose())
                source.allow_next.set()
            else:
                source.allow_next.set()
                close_task = asyncio.create_task(response.aclose())

            next_result, close_result = await asyncio.wait_for(
                asyncio.gather(pending_next, close_task, return_exceptions=True),
                timeout=1,
            )
            self.assertIsNone(close_result)
            self.assertTrue(
                next_result == index
                or isinstance(next_result, (asyncio.CancelledError, StopAsyncIteration))
            )
            self.assertEqual(source.close_calls, 1)
            self.assertEqual(source.release_count, 1)

    async def test_natural_completion_closes_registered_dependency(self):
        dependency = CloseTrackedAsyncIterator([])

        async def source():
            yield "first"

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )

        self.assertEqual([item async for item in response], ["first"])
        await response.aclose()

        self.assertEqual(dependency.close_calls, 1)
        self.assertEqual(dependency.release_count, 1)
        self.assertIs(response._close_state, _CloseState.CLOSED)

    async def test_iteration_error_closes_dependency_without_sticky_failure(self):
        dependency = CloseTrackedAsyncIterator([])
        iteration_error = RuntimeError("backend failed")

        async def source():
            yield "first"
            raise iteration_error

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )

        self.assertEqual(await response.__anext__(), "first")
        with self.assertRaises(RuntimeError) as raised:
            await response.__anext__()
        self.assertIs(raised.exception, iteration_error)
        await response.aclose()

        self.assertEqual(dependency.close_calls, 1)
        self.assertEqual(dependency.release_count, 1)
        self.assertEqual(response._sticky_close_errors, [])
        self.assertIs(response._close_state, _CloseState.CLOSED)

    async def test_iteration_cancellation_closes_dependency_without_sticky_failure(self):
        dependency = CloseTrackedAsyncIterator([])
        iteration_error = asyncio.CancelledError("backend cancelled")

        async def source():
            yield "first"
            raise iteration_error

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )

        self.assertEqual(await response.__anext__(), "first")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await response.__anext__()
        self.assertIs(raised.exception, iteration_error)
        await response.aclose()

        self.assertEqual(dependency.close_calls, 1)
        self.assertEqual(dependency.release_count, 1)
        self.assertEqual(response._sticky_close_errors, [])
        self.assertIs(response._close_state, _CloseState.CLOSED)

    async def test_iteration_error_yields_to_cleanup_failure_then_retry_succeeds(self):
        dependency = FailOnceCloseAsyncIterator([])
        iteration_error = ValueError("backend failed")

        async def source():
            yield "first"
            raise iteration_error

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )

        self.assertEqual(await response.__anext__(), "first")
        with self.assertRaisesRegex(RuntimeError, "injected close failure") as raised:
            await response.__anext__()
        self.assertIs(raised.exception.__cause__, iteration_error)
        self.assertIs(response._close_state, _CloseState.CLOSE_FAILED)

        await response.aclose()
        self.assertEqual(dependency.close_calls, 2)
        self.assertEqual(dependency.release_count, 1)
        self.assertIs(response._close_state, _CloseState.CLOSED)

    async def test_managed_close_failure_is_retried_after_wrapper_terminates(self):
        dependency = FailOnceCloseAsyncIterator([])
        registry = CloseDependencyRegistry()
        managed_dependency = registry.wrap(dependency)

        async def source():
            try:
                yield "first"
            finally:
                await managed_dependency.aclose()

        source_generator = source()
        response = CompleteResponseAsyncGenerator(
            source_generator, collect_last, close_dependencies=registry
        )
        self.assertEqual(await response.__anext__(), "first")

        await response.aclose()

        self.assertIsNone(source_generator.ag_frame)
        self.assertEqual(dependency.close_calls, 2)
        self.assertEqual(dependency.release_count, 1)
        self.assertEqual(response._sticky_close_errors, [])
        self.assertIs(response._close_state, _CloseState.CLOSED)

    async def test_unrelated_root_close_error_remains_sticky(self):
        dependency = CloseTrackedAsyncIterator([])
        close_error = RuntimeError("wrapper close failed")

        async def source():
            try:
                yield "first"
            finally:
                raise close_error

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )
        self.assertEqual(await response.__anext__(), "first")

        for _ in range(2):
            with self.assertRaises(RuntimeError) as raised:
                await response.aclose()
            self.assertIs(raised.exception, close_error)

        self.assertEqual(dependency.close_calls, 1)
        self.assertEqual(dependency.release_count, 1)
        self.assertEqual(response._sticky_close_errors, [close_error])
        self.assertIs(response._close_state, _CloseState.CLOSE_FAILED)

    async def test_native_dependency_terminal_close_failure_stays_failed(self):
        close_error = RuntimeError("native dependency close failed")

        async def dependency_source():
            try:
                yield "dependency"
            finally:
                raise close_error

        dependency = dependency_source()
        self.assertEqual(await dependency.__anext__(), "dependency")
        registry = CloseDependencyRegistry()
        managed_dependency = registry.wrap(dependency)

        async def source():
            try:
                yield "first"
            finally:
                await managed_dependency.aclose()

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=registry
        )
        self.assertEqual(await response.__anext__(), "first")

        first_marker = None
        for _ in range(2):
            with self.assertRaises(_ManagedCloseError) as raised:
                await response.aclose()
            if first_marker is None:
                first_marker = raised.exception
            else:
                self.assertIs(raised.exception, first_marker)
            self.assertIs(raised.exception.error, close_error)

        entry = managed_dependency._entry
        self.assertIsNone(dependency.ag_frame)
        self.assertFalse(entry.closed)
        self.assertIs(entry._terminal_close_error, close_error)
        self.assertIs(response._close_state, _CloseState.CLOSE_FAILED)

    async def test_external_close_sticks_active_cleanup_error(self):
        dependency = CloseTrackedAsyncIterator([])
        next_started = asyncio.Event()
        close_error = RuntimeError("backend cleanup failed during cancellation")

        async def source():
            try:
                next_started.set()
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                raise close_error

        response = CompleteResponseAsyncGenerator(
            source(), collect_last, close_dependencies=[dependency]
        )
        pending_next = asyncio.create_task(response.__anext__())
        await next_started.wait()

        with self.assertRaises(RuntimeError) as raised:
            await response.aclose()
        self.assertIs(raised.exception, close_error)

        with self.assertRaises(RuntimeError) as raised_next:
            await pending_next
        self.assertIs(raised_next.exception, close_error)
        with self.assertRaises(RuntimeError) as raised_again:
            await response.aclose()
        self.assertIs(raised_again.exception, close_error)
        self.assertEqual(dependency.close_calls, 1)
        self.assertEqual(dependency.release_count, 1)
        self.assertEqual(response._sticky_close_errors, [close_error])
        self.assertIs(response._close_state, _CloseState.CLOSE_FAILED)


if __name__ == "__main__":
    main()
