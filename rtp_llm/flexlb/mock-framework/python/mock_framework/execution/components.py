"""Common component files, readiness, and registration; native configuration belongs to its caller."""
import asyncio
import copy
import json
import os
from ..config.validation import duration
from .records import write_json
from .transport import ExecutionError


class ComponentLauncher:
    def __init__(self, supervisor, topology, records, environment):
        self.supervisor, self.topology, self.records = supervisor, topology, records
        self.environment = environment

    async def start(self, role, index, recipe, *, configure=None, validate=None, require_endpoints=False, attempt=0):
        name = f'{role}-{index}'
        directory = self.records.directory / 'components' / name
        process = name if attempt == 0 else f'{name}-restart-{attempt}'
        if attempt:
            directory = directory / f'restart-{attempt}'
        directory.mkdir(parents=True)
        config, ready = directory / 'config.json', directory / 'ready.json'
        write_json(config, {'component_role': role, 'component': copy.deepcopy(self.environment[role]),
                           'environment': self.environment, 'index': index, 'id': name})
        env = {} if configure is None else configure(directory)
        await self.supervisor.spawn(process, recipe['command'], recipe['cwd'],
            {**os.environ, **recipe['env'], **env, 'MOCK_TOPOLOGY_FILE': str(self.topology.directory / 'topology.json'),
             'MOCK_NODE_ID': name, 'MOCK_COMPONENT': role, 'MOCK_CONFIG_FILE': str(config),
             'MOCK_READY_FILE': str(ready), 'MOCK_OUTPUT_DIR': str(directory)}, directory)
        async def await_ready():
            while not ready.exists():
                await asyncio.sleep(.02)
            value = json.loads(ready.read_text())
            if set(value) != {'url'} or not isinstance(value['url'], str):
                raise ExecutionError(f'Invalid readiness file for {name}')
            return value['url']
        control = await self.supervisor.guard(await_ready(), float(duration(recipe['startup_timeout'])))
        self.records.emit(event='process_ready', node=process, url=control)
        path = directory / 'endpoints.json'
        endpoint = json.loads(path.read_text()) if path.is_file() else None
        if endpoint is None and require_endpoints:
            raise ExecutionError('Native Master did not publish endpoints')
        if endpoint is not None and validate is not None:
            validate(endpoint, control)
        self.topology.register(name, role, control, endpoint)
        return endpoint
