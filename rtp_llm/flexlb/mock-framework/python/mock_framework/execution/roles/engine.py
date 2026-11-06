"""Physical Engine placement; Case operations address instances, never JVMs."""
import asyncio
import copy
import json
import os
import socket
from ...config.validation import duration
from ..run_trace import write_json
from .. import ExecutionError

ROLES = ('prefill', 'decode', 'vit')


class EngineRole:
    def __init__(self, *, supervisor, topology, records, environment, recipes, monitoring):
        self.supervisor, self.topology, self.records = supervisor, topology, records
        self.environment, self.recipes, self.monitoring = environment, recipes, monitoring
        self.instances, self.counters = set(), {}
        self.processes = set()
        self.used_ports = set()
        self.lock = asyncio.Lock()

    def _port(self):
        # Binding is a probe, not a reservation; a subsequent bind failure fails the run.
        for _ in range(32):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise ExecutionError('Cannot allocate a fresh Engine listener port')

    async def _start_instance(self, role, node, directory):
        recipe = self.recipes[role]
        rpc_port, control_port = self._port(), self._port()
        ready = directory / 'ready.json'
        self.processes.add(node)
        child = await self.supervisor.spawn(node, recipe['command'], recipe['cwd'],
            {**os.environ, **recipe['env'], 'MOCK_BIND_HOST': '127.0.0.1', 'MOCK_ADVERTISE_HOST': '127.0.0.1',
             'MOCK_CONFIG_FILE': str(directory / 'config.json'), 'MOCK_OUTPUT_DIR': str(directory),
             'MOCK_READY_FILE': str(ready), 'START_PORT': str(control_port), 'MOCK_RPC_PORT': str(rpc_port)}, directory)

        async def read_ready():
            while not ready.exists():
                await asyncio.sleep(.02)
            control = f'http://127.0.0.1:{control_port}'
            rpc = f'127.0.0.1:{rpc_port}'
            endpoint = json.loads((directory / 'endpoints.json').read_text())
            expected = {'id': node, 'role': role, 'rpc': rpc, 'control': control}
            if json.loads(ready.read_text()) != {'url': control} or any(
                    endpoint.get(key) != value for key, value in expected.items()):
                raise ExecutionError('Engine endpoints differ from the assigned identity and listeners')
            return endpoint

        endpoint = await self.supervisor.guard(read_ready(), float(duration(recipe['startup_timeout'])))
        self.instances.add(node)
        self.topology.register(node, role, endpoint['control'], endpoint)
        self.records.emit(event='process_ready', node=node, url=endpoint['control'], role=role, instance_id=node, generation=0)
        self.records.emit(event='instance_started', node=node, pid=child.pid, rpc=endpoint['rpc'],
                          role=role, instance_id=node, generation=0)
        return endpoint['rpc']

    async def scale_out(self, role, count):
        async with self.lock:
            if role not in ROLES or role not in self.environment or not 1 <= count <= 1024:
                raise ExecutionError('scale_out requires a configured Engine role and count in 1..1024')
            result = []
            for _ in range(count):
                index = self.counters.get(role, 0)
                self.counters[role] = index + 1
                node = f'{role}-{index}'
                directory = self.records.directory / 'components' / node
                directory.mkdir(parents=True)
                component = copy.deepcopy(self.environment[role])
                write_json(directory / 'config.json', {'component_role': role, 'component': component,
                    'environment': self.environment, 'index': index, 'id': node})
                result.append(await self._start_instance(role, node, directory))
                if self.monitoring:
                    self.monitoring.discover(node, directory)
            if self.monitoring and self.monitoring.baselined:
                await self.monitoring.baseline([item['id'] for item in self.topology.active() if item['rpc'] in result])
            return result

    async def scale_in(self, role, targets, count):
        async with self.lock:
            candidates = [item for item in self.topology.active() if item['role'] == role and item['id'] in self.instances]
            known = {item['rpc']: item for item in candidates}
            if targets is not None:
                if not targets or len(set(targets)) != len(targets) or set(targets) - known.keys():
                    raise ExecutionError('Engine targets must be nonempty, unique, active addresses of the selected role')
                candidates = [known[address] for address in targets]
            if not candidates or count is not None and not 1 <= count <= len(candidates):
                raise ExecutionError('No Engine candidates or scale_in count exceeds the selected set')
            selected = candidates if count is None else candidates[:count]
            # Selection is fixed once, before stopping any process.
            self.records.emit(event='instances_selected', operation='scale_in', targets=[x['rpc'] for x in selected])
            if self.monitoring:
                await self.monitoring.final_scrape([item['id'] for item in selected])
            for item in selected:
                node = item['id']
                errors = await self.supervisor.stop_intentionally(node)
                if errors:
                    raise ExecutionError('; '.join(errors))
                (self.records.directory / 'components' / node / 'ready.json').unlink(missing_ok=True)
                self.processes.remove(node)
                self.instances.remove(node)
                self.topology.retire(node)
                if self.monitoring:
                    self.monitoring.retire(node)
                self.records.emit(event='instance_stopped', node=node, rpc=item['rpc'],
                                  role=role, instance_id=node, generation=0)
            return [item['rpc'] for item in selected]

    async def start(self):
        for role, recipe in self.recipes.items():
            if role in ROLES:
                count = self.environment[role]['deployment']['replicas']
                await self.supervisor.guard(self.scale_out(role, count), float(duration(recipe['startup_timeout'])) * count)

    async def close(self):
        names = [name for name, _ in self.supervisor.children if name in self.processes]
        errors = await self.supervisor.stop(names, 3)
        active = {name for name, _ in self.supervisor.children}
        for node in self.processes - active:
            (self.records.directory / 'components' / node / 'ready.json').unlink(missing_ok=True)
        return errors
