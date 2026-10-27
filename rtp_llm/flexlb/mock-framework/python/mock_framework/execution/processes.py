import asyncio
import json
import os
from pathlib import Path
import signal

from ..config.values import duration, fields, text
from .records import write_json
from .transport import ExecutionError


def prepare_launch(raw, case, base):
    """Launch recipes are developer bindings, separate from case business configuration."""
    fields(raw, {'components', 'bindings'}, {'components', 'bindings'}, where='launch')
    if not isinstance(raw['components'], dict) or not isinstance(raw['bindings'], dict):
        raise ExecutionError('launch components/bindings must be mappings')
    components = {}
    for role, recipe in raw['components'].items():
        if role not in case.environment or role == 'compute':
            raise ExecutionError(f'Launch recipe names an unconfigured component: {role}')
        fields(recipe, {'command', 'cwd', 'env', 'startup_timeout'}, {'command'}, where='launch.' + role)
        command = recipe['command']
        if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
            raise ExecutionError('command must be a nonempty argv list; no shell expansion')
        env = recipe.get('env', {})
        if not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()):
            raise ExecutionError('Launch env must contain string keys and values')
        if any(k.startswith('MOCK_') for k in env):
            raise ExecutionError('MOCK_* launch variables are assigned by the runner')
        startup = recipe.get('startup_timeout', '30s')
        duration(startup, positive=True)
        cwd = Path(base) / text(recipe.get('cwd', '.'), 'launch.cwd')
        if not cwd.is_dir():
            raise ExecutionError(f'Launch cwd is not a directory: {cwd}')
        deployment = case.environment[role]['deployment']
        if deployment.get('resources') or deployment.get('heap'):
            raise ExecutionError('This local launcher does not yet enforce resources/heap; refusing to ignore them')
        components[role] = {**recipe, 'command': command, 'env': env, 'cwd': str(cwd.resolve()),
                            'startup_timeout': startup}
    missing = set(case.environment) - {'compute'} - set(components)
    if missing:
        raise ExecutionError(f'Missing component launch recipes: {sorted(missing)}')
    ids = {f'{role}-{n}' for role in components for n in range(case.environment[role]['deployment']['replicas'])}
    for prefix, node in raw['bindings'].items():
        text(prefix, 'binding prefix')
        text(node, 'binding process id')
        if not prefix.endswith('/') or node not in ids:
            raise ExecutionError('Bindings need a route prefix ending in / and an existing process id')
    return {'components': components, 'bindings': dict(raw['bindings'])}


class Processes:
    def __init__(self, case, launch, records):
        self.case, self.launch, self.records = case, launch, records
        self.children, self.monitors, self.urls = [], [], {}
        self.failure = asyncio.get_running_loop().create_future()
        self.closing = False

    def endpoint(self, interface):
        prefixes = [p for p in self.launch['bindings'] if interface.startswith(p)]
        if not prefixes:
            raise ExecutionError(f'No process binding for {interface}')
        return self.urls[self.launch['bindings'][max(prefixes, key=len)]]

    async def _monitor(self, node, child):
        code = await child.wait()
        if not self.closing and not self.failure.done():
            self.failure.set_result(f'Component {node} exited unexpectedly with code {code}')

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

    async def start(self):
        for role, recipe in self.launch['components'].items():
            for index in range(self.case.environment[role]['deployment']['replicas']):
                node = f'{role}-{index}'
                directory = self.records.directory / 'components' / node
                directory.mkdir(parents=True)
                config = directory / 'config.json'
                ready = directory / 'ready.json'
                write_json(config, {'component_role': role, 'component': self.case.environment[role],
                                    'environment': self.case.environment, 'index': index, 'id': node})
                env = {**os.environ, **recipe['env'], 'MOCK_NODE_ID': node, 'MOCK_COMPONENT': role,
                       'MOCK_CONFIG_FILE': str(config), 'MOCK_READY_FILE': str(ready), 'MOCK_OUTPUT_DIR': str(directory)}
                with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
                    spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                        *recipe['command'], cwd=recipe['cwd'], env=env, stdout=out, stderr=err, start_new_session=True))
                    try:
                        child = await asyncio.shield(spawning)
                    except asyncio.CancelledError:
                        child = await spawning
                        self.children.append((node, child))
                        self.monitors.append(asyncio.create_task(self._monitor(node, child)))
                        self.records.emit(event='process_started', node=node, pid=child.pid)
                        raise
                self.children.append((node, child))
                self.monitors.append(asyncio.create_task(self._monitor(node, child)))
                self.records.emit(event='process_started', node=node, pid=child.pid)
                async def await_ready():
                    while not ready.exists():
                        await asyncio.sleep(.02)
                    value = json.loads(ready.read_text(encoding='utf-8'))
                    if set(value) != {'url'} or not isinstance(value['url'], str):
                        raise ExecutionError(f'Invalid readiness file for {node}')
                    self.urls[node] = value['url']
                await self.guard(await_ready(), float(duration(recipe['startup_timeout'])))
                self.records.emit(event='process_ready', node=node, url=self.urls[node])

    async def close(self):
        self.closing = True
        errors = []
        # Signal the entire owned process group, including grandchildren, even if its leader exited.
        for node, child in reversed(self.children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{node}: {error}')
        # A shared grace period avoids multiplying teardown time by replica count.
        deadline = asyncio.get_running_loop().time() + 3
        while any(child.returncode is None for _, child in self.children) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.02)
        for node, child in reversed(self.children):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{node}: {error}')
            try:
                await asyncio.wait_for(child.wait(), timeout=3)
            except TimeoutError:
                errors.append(f'{node}: process did not exit after SIGKILL')
            self.records.emit(event='process_stopped', node=node, code=child.returncode)
        for monitor in self.monitors:
            if not monitor.done():
                monitor.cancel()
        await asyncio.gather(*self.monitors, return_exceptions=True)
        return errors
