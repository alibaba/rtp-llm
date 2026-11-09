"""Own OS processes and interrupt operations when one exits unexpectedly."""
import asyncio
import json
import os
import signal
from ..config.validation import duration
from .step_trace import write_json
from . import ExecutionError


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

    def _process_event(self, event, name, **fields):
        instance_id, _, generation = name.partition('-restart-')
        role = instance_id.split('-', 1)[0]
        if role in ('master', 'prefill', 'decode', 'vit', 'client', 'zookeeper', 'prometheus'):
            fields.update(role=role, instance_id=instance_id, generation=int(generation or '0'))
        self.records.emit(event=event, node=name, **fields)

    async def spawn(self, name, command, cwd, env, directory, *, describe=None, zero_exit=False):
        if name in self._children or name in self._expected:
            raise ExecutionError(f'Process identity has already been used: {name}')
        def retain(child):
            self._children[name] = child
            if zero_exit:
                self._zero_exit.add(name)
            self._monitors.append(asyncio.create_task(self._monitor(name, child, describe)))
            self._process_event('process_started', name, pid=child.pid)
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

    async def stop_intentionally(self, name, *, mode='terminate', grace=3):
        child = self._children[name]
        if child.returncode is not None:
            raise ExecutionError(f'{name} already exited before intentional {mode}')
        if mode not in ('terminate', 'kill'):
            raise ExecutionError(f'Unknown intentional stop mode: {mode}')
        self._expected.add(name)
        if mode == 'terminate':
            return await self.stop([name], grace)
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
            self._process_event('process_stopped', name, code=child.returncode)
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
            self._process_event('process_stopped', name, code=child.returncode)
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


async def launch_component(supervisor, records, environment, role, index, recipe, *, configure=None, attempt=0):
    name = f'{role}-{index}'
    directory = records.directory / 'components' / name
    process = name if attempt == 0 else f'{name}-restart-{attempt}'
    if attempt:
        directory = directory / f'restart-{attempt}'
    directory.mkdir(parents=True)
    config, ready = directory / 'config.json', directory / 'ready.json'
    write_json(config, {'component_role': role, 'component': environment[role],
                       'environment': environment, 'index': index, 'id': name})
    env = {} if configure is None else configure(directory)
    child = await supervisor.spawn(process, recipe['command'], recipe['cwd'],
        {**os.environ, **recipe['env'], **env, 'MOCK_TOPOLOGY_FILE': str(records.directory / 'topology.json'),
         'MOCK_NODE_ID': name, 'MOCK_CONFIG_FILE': str(config),
         'MOCK_READY_FILE': str(ready), 'MOCK_OUTPUT_DIR': str(directory)}, directory)
    async def await_ready():
        while not ready.exists():
            await asyncio.sleep(.02)
        value = json.loads(ready.read_text())
        if set(value) != {'url'} or not isinstance(value['url'], str):
            raise ExecutionError(f'Invalid readiness file for {name}')
        return value['url']
    control = await supervisor.guard(await_ready(), float(duration(recipe['startup_timeout'])))
    records.emit(event='process_ready', node=process, url=control, role=role,
                 instance_id=name, generation=attempt)
    path = directory / 'endpoints.json'
    endpoint = json.loads(path.read_text()) if path.is_file() else None
    return child, control, endpoint
