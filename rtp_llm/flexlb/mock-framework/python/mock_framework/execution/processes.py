"""Own OS processes and interrupt operations when one exits unexpectedly."""
import asyncio
import os
import signal
from .transport import ExecutionError


class ProcessSupervisor:
    def __init__(self, records):
        self.records = records
        self._children, self._monitors = {}, []
        self._expected, self._zero_exit = set(), set()
        self.failure = asyncio.get_running_loop().create_future()
        self.closing = False

    @property
    def children(self):
        return tuple(self._children.items())

    def child(self, name):
        return self._children[name]

    def fail(self, message):
        if not self.closing and not self.failure.done():
            self.failure.set_result(message)

    async def _monitor(self, name, child, describe):
        code = await child.wait()
        if name not in self._expected:
            detail = '' if describe is None else '; ' + describe()
            self.fail(f'Component {name} exited unexpectedly with code {code}{detail}')

    async def guard(self, operation, timeout):
        task = asyncio.create_task(operation)
        try:
            async with asyncio.timeout(timeout):
                done, _ = await asyncio.wait((task, self.failure), return_when=asyncio.FIRST_COMPLETED)
                if self.failure in done:
                    raise ExecutionError(self.failure.result())
                return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def spawn(self, name, command, cwd, env, directory, *, describe=None, zero_exit=False):
        if name in self._children or name in self._expected:
            raise ExecutionError(f'Process identity has already been used: {name}')
        def retain(child):
            self._children[name] = child
            if zero_exit:
                self._zero_exit.add(name)
            self._monitors.append(asyncio.create_task(self._monitor(name, child, describe)))
            self.records.emit(event='process_started', node=name, pid=child.pid)
        with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                *command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True))
            try:
                child = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                retain(await spawning)
                raise
        retain(child)
        return child

    async def stop_expected(self, name, grace=3):
        child = self._children[name]
        if child.returncode is not None:
            raise ExecutionError(f'{name} already exited before intentional stop')
        self._expected.add(name)
        errors = await self.stop([name], grace)
        return errors

    async def kill_expected(self, name):
        child = self._children[name]
        if child.returncode is not None:
            raise ExecutionError(f'{name} already exited before intentional kill')
        self._expected.add(name)
        errors = []
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except OSError as error:
            errors.append(f'{name}: cannot kill process group: {error}')
        try:
            await asyncio.wait_for(child.wait(), timeout=3)
        except TimeoutError:
            errors.append(f'{name}: process did not exit after intentional SIGKILL')
        if child.returncode is not None:
            if child.returncode not in (-signal.SIGKILL, 128 + signal.SIGKILL):
                errors.append(f'{name}: unexpected exit status during kill: {child.returncode}')
            self.records.emit(event='process_stopped', node=name, code=child.returncode)
            del self._children[name]
        return errors

    async def stop(self, names, grace):
        children = [(name, self._children[name]) for name in names if name in self._children]
        errors = [f'{name}: exited unexpectedly before stop with code {child.returncode}'
                  for name, child in children if child.returncode is not None and name not in self._expected]
        self._expected.update(name for name, _ in children)
        # Kill only owned process groups, including grandchildren after their leader exits.
        for name, child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{name}: {error}')
        deadline = asyncio.get_running_loop().time() + grace
        while any(child.returncode is None for _, child in children) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.02)
        for name, child in reversed(children):
            killed = False
            try:
                os.killpg(child.pid, signal.SIGKILL)
                killed = child.returncode is None
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{name}: {error}')
            try:
                await asyncio.wait_for(child.wait(), timeout=3)
            except TimeoutError:
                errors.append(f'{name}: process did not exit after SIGKILL')
            if killed:
                errors.append(f'{name}: did not stop within the grace period; SIGKILL was required')
            if name in self._expected and child.returncode not in (0, -signal.SIGTERM, 128 + signal.SIGTERM):
                errors.append(f'{name}: unexpected exit status during stop: {child.returncode}')
            if name in self._zero_exit and (killed or child.returncode != 0):
                errors.append(f'{name}: did not stop gracefully; retained data may be incomplete')
            self.records.emit(event='process_stopped', node=name, code=child.returncode)
            if child.returncode is not None:
                del self._children[name]
        return errors

    async def close(self):
        self.closing = True
        errors = await self.stop(list(self._children), 3)
        for monitor in self._monitors:
            if not monitor.done():
                monitor.cancel()
        await asyncio.gather(*self._monitors, return_exceptions=True)
        return errors
