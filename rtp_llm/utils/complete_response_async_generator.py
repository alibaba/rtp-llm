import asyncio
import inspect
import threading
from enum import Enum, auto
from typing import Any, AsyncGenerator, Callable, Iterable


class _CloseState(Enum):
    OPEN = auto()
    CLOSING = auto()
    CLOSE_FAILED = auto()
    CLOSED = auto()


class _CloseEntry:
    def __init__(self, dependency: Any) -> None:
        self.dependency = dependency
        self._state_lock = threading.Lock()
        self._close_task = None
        self._terminal_close_error = None
        self.closed = False

    async def aclose(self) -> None:
        with self._state_lock:
            if self.closed:
                return
            if self._terminal_close_error is not None:
                raise self._terminal_close_error
            if self._close_task is None:
                close_task = asyncio.create_task(self._close_dependency())
                close_task.add_done_callback(_consume_task_exception)
                self._close_task = close_task
            else:
                close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_dependency(self) -> None:
        close_task = asyncio.current_task()
        try:
            await self.dependency.aclose()
        except BaseException as e:
            with self._state_lock:
                if self._close_task is close_task:
                    self._close_task = None
                if (
                    inspect.isasyncgen(self.dependency)
                    and self.dependency.ag_frame is None
                ):
                    self._terminal_close_error = e
            raise
        else:
            with self._state_lock:
                if self._close_task is close_task:
                    self._close_task = None
                    self.closed = True


class _ManagedCloseError(RuntimeError):
    def __init__(self, entry: _CloseEntry, error: BaseException) -> None:
        super().__init__(str(error))
        self.entry = entry
        self.error = error


class _ManagedCloseDependency:
    def __init__(self, entry: _CloseEntry, accepted: bool = True) -> None:
        self._entry = entry
        self.accepted = accepted

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self._entry.dependency.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        try:
            await self._entry.aclose()
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            raise _ManagedCloseError(self._entry, e) from e

    def __getattr__(self, name):
        return getattr(self._entry.dependency, name)


def _consume_task_exception(close_task) -> None:
    if close_task.cancelled():
        return
    try:
        close_task.exception()
    except BaseException:
        pass


def _recovered_managed_close(error: BaseException | None) -> bool:
    return isinstance(error, _ManagedCloseError) and error.entry.closed


class CloseDependencyRegistry:
    def __init__(self, dependencies: Iterable[Any] = ()) -> None:
        self._state_lock = threading.Lock()
        self._entries = []
        self._accepting = True
        self._close_state = _CloseState.OPEN
        self._close_task = None
        for dependency in dependencies:
            if not self.add(dependency):
                raise RuntimeError("cannot add a dependency to a closed registry")

    def add(self, dependency: Any) -> bool:
        with self._state_lock:
            if not self._accepting:
                return False
            if any(entry.dependency is dependency for entry in self._entries):
                return True
            self._entries.append(_CloseEntry(dependency))
            return True

    def wrap(self, dependency: Any):
        with self._state_lock:
            accepted = self._accepting
            for entry in self._entries:
                if entry.dependency is dependency:
                    return _ManagedCloseDependency(entry, accepted)
            entry = _CloseEntry(dependency)
            self._entries.append(entry)
            if not accepted and self._close_state is _CloseState.CLOSED:
                self._close_state = _CloseState.CLOSE_FAILED
            return _ManagedCloseDependency(entry, accepted)

    async def aclose(self) -> None:
        with self._state_lock:
            if self._close_state is _CloseState.CLOSED:
                return
            if self._close_task is None:
                self._accepting = False
                self._close_state = _CloseState.CLOSING
                close_task = asyncio.create_task(self._close_dependencies())
                close_task.add_done_callback(_consume_task_exception)
                self._close_task = close_task
            else:
                close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_dependencies(self) -> None:
        close_task = asyncio.current_task()
        attempted = set()
        failures = []
        while True:
            with self._state_lock:
                entry = next(
                    (
                        item
                        for item in self._entries
                        if not item.closed and id(item) not in attempted
                    ),
                    None,
                )
            if entry is None:
                break
            attempted.add(id(entry))
            try:
                await entry.aclose()
            except BaseException as e:
                failures.append((entry, e))

        with self._state_lock:
            all_closed = all(entry.closed for entry in self._entries)
            if self._close_task is close_task:
                self._close_task = None
                if all_closed:
                    self._close_state = _CloseState.CLOSED
                else:
                    self._close_state = _CloseState.CLOSE_FAILED

        for entry, error in failures:
            if not entry.closed:
                raise error
        if not all_closed:
            raise RuntimeError("one or more close dependencies remain open")


class CompleteResponseAsyncGenerator:
    def __init__(
        self,
        generator: AsyncGenerator,
        collect_complete_response_func: Callable,
        close_dependencies: Iterable[Any] = (),
    ):
        self._generator = generator
        self._collect_complete_response_func = collect_complete_response_func
        self._close_dependencies = (
            close_dependencies
            if isinstance(close_dependencies, CloseDependencyRegistry)
            else CloseDependencyRegistry(close_dependencies)
        )
        self._all_responses = []
        self._state_lock = threading.Lock()
        self._close_state = _CloseState.OPEN
        self._close_task = None
        self._active_next_task = None
        self._active_next_done = None
        self._sticky_close_errors = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        current_task = asyncio.current_task()
        with self._state_lock:
            if self._close_state is not _CloseState.OPEN:
                raise StopAsyncIteration
            if self._active_next_task is not None:
                raise RuntimeError("anext(): asynchronous generator is already running")
            self._active_next_task = current_task
            self._active_next_done = asyncio.get_running_loop().create_future()

        exhausted = False
        next_error = None
        try:
            response = await self._generator.__anext__()
            self._all_responses.append(response)
            return response
        except StopAsyncIteration:
            exhausted = True
        except BaseException as e:
            next_error = e
        finally:
            with self._state_lock:
                if self._active_next_task is current_task:
                    self._active_next_task = None
                    active_next_done = self._active_next_done
                    self._active_next_done = None
                    if active_next_done is not None and not active_next_done.done():
                        active_next_done.set_result(next_error)

        if exhausted:
            await self.aclose()
            raise StopAsyncIteration
        if next_error is not None:
            if isinstance(next_error, _ManagedCloseError):
                self._remember_terminal_close_error(next_error)
            with self._state_lock:
                close_already_started = self._close_task is not None
            if close_already_started:
                raise next_error
            try:
                await self.aclose()
            except BaseException as close_error:
                if close_error is next_error:
                    raise
                raise close_error from next_error
            raise next_error

    async def aclose(self):
        with self._state_lock:
            if self._close_state is _CloseState.CLOSED:
                if self._close_task is None:
                    return None
                close_task = self._close_task
            elif self._close_task is not None:
                close_task = self._close_task
            else:
                self._close_state = _CloseState.CLOSING
                close_task = asyncio.create_task(self._close_generator())
                close_task.add_done_callback(_consume_task_exception)
                self._close_task = close_task

        return await asyncio.shield(close_task)

    async def _close_generator(self):
        close_task = asyncio.current_task()
        try:
            with self._state_lock:
                active_next_task = self._active_next_task
                active_next_done = self._active_next_done
                if active_next_task is not None:
                    active_next_task.cancel()
            if active_next_done is not None:
                active_next_error = await active_next_done
            else:
                active_next_error = None
            result = None
            close_errors = list(self._sticky_close_errors)
            if active_next_error is not None and not isinstance(
                active_next_error,
                (asyncio.CancelledError, StopAsyncIteration, GeneratorExit),
            ):
                close_errors.append(active_next_error)
                self._remember_terminal_close_error(active_next_error)
            try:
                result = await self._generator.aclose()
            except BaseException as e:
                close_errors.append(e)
                self._remember_terminal_close_error(e)
            try:
                await self._close_dependencies.aclose()
            except BaseException as e:
                close_errors.append(e)

            self._sticky_close_errors = [
                error
                for error in self._sticky_close_errors
                if not _recovered_managed_close(error)
            ]
            close_errors = [
                error
                for error in close_errors
                if not _recovered_managed_close(error)
            ]
            if close_errors:
                raise close_errors[0]
        except BaseException:
            with self._state_lock:
                if self._close_task is close_task:
                    self._close_task = None
                    self._close_state = _CloseState.CLOSE_FAILED
            raise
        else:
            with self._state_lock:
                if self._close_task is close_task:
                    self._close_state = _CloseState.CLOSED
                    self._close_task = None
            return result

    def _remember_terminal_close_error(self, error: BaseException) -> None:
        if not inspect.isasyncgen(self._generator) or self._generator.ag_frame is not None:
            return
        if any(item is error for item in self._sticky_close_errors):
            return
        self._sticky_close_errors.append(error)

    async def gen_complete_response_once(self) -> Any:
        return await self._collect_complete_response_func(
            CompleteResponseAsyncGenerator.generate_from_list(self._all_responses)
        )

    @staticmethod
    async def generate_from_list(response_list) -> AsyncGenerator:
        for response in response_list:
            yield response

    @staticmethod
    async def get_last_value(all_responses: AsyncGenerator):
        response = None
        try:
            async for response in all_responses:
                pass
        except StopAsyncIteration:
            pass
        return response
