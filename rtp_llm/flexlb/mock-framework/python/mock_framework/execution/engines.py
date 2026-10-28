"""Physical Engine placement; Case operations address instances, never JVMs."""
import asyncio
import copy
import json
import os
import socket
from ..config.values import duration
from .records import timestamp, write_json
from .transport import post, ExecutionError

ROLES = ('prefill', 'decode', 'vit')


class EngineHosts:
    def __init__(self, owner):
        self.owner = owner
        self.hosts, self.instances, self.counters = {}, {}, {}
        self.used_ports = set()
        self.lock = asyncio.Lock()

    def port(self):
        # Binding is a probe, not a reservation; a subsequent bind failure fails the run.
        for _ in range(32):
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise ExecutionError('Cannot allocate a fresh Engine listener port')

    async def call(self, host, operation, args):
        async with asyncio.timeout(10):
            return await post(host['url'], {'interface': 'host/' + operation, 'args': args,
                'context': {'run': 'foreground', 'query_time': timestamp(), 'observation': None, 'timeout_ms': 9000}})

    async def create(self, role, count):
        async with self.lock:
            if role not in ROLES or role not in self.owner.case.environment or not 1 <= count <= 1024:
                raise ExecutionError('scale_out requires a configured Engine role and count in 1..1024')
            result = []
            for _ in range(count):
                index = self.counters.get(role, 0)
                self.counters[role] = index + 1
                node = f'{role}-{index}'
                directory = self.owner.records.directory / 'components' / node
                directory.mkdir(parents=True)
                component = copy.deepcopy(self.owner.case.environment[role])
                write_json(directory / 'config.json', {'component_role': role, 'component': component,
                    'environment': self.owner.case.environment, 'index': index, 'id': node})
                host_name = 'engine-shared' if self.owner.launch['engine_mode'] == 'shared' else 'engine-' + node
                host = self.hosts.get(host_name)
                if host is None:
                    recipe = self.owner.launch['components'][role]
                    host_dir = self.owner.records.directory / 'hosts' / host_name
                    host_dir.mkdir(parents=True)
                    ready = host_dir / 'ready.json'
                    host_port = self.port()
                    child = await self.owner.spawn(host_name, recipe['command'], recipe['cwd'],
                        {**os.environ, **recipe['env'], 'MOCK_COMPONENTS_DIR': str(directory.parent),
                         'MOCK_OUTPUT_DIR': str(host_dir), 'MOCK_READY_FILE': str(ready), 'MOCK_HOST_PORT': str(host_port)}, host_dir)
                    host = {'child': child, 'members': set(), 'url': None}
                    self.hosts[host_name] = host
                    async def await_ready():
                        while not ready.exists():
                            await asyncio.sleep(.02)
                        value = json.loads(ready.read_text())
                        if value != {'url': f'http://127.0.0.1:{host_port}'}:
                            raise ExecutionError('Invalid Engine host readiness')
                        return value['url']
                    host['url'] = await self.owner.guard(await_ready(), float(duration(recipe['startup_timeout'])))
                    self.owner.records.emit(event='process_ready', node=host_name, url=host['url'])
                rpc_port, control_port = self.port(), self.port()
                endpoint = await self.call(host, 'create', {'id': node, 'rpcPort': rpc_port, 'controlPort': control_port})
                saved = json.loads((directory / 'endpoints.json').read_text())
                expected = {'id': node, 'role': role, 'rpc': f'127.0.0.1:{rpc_port}',
                    'control': f'http://127.0.0.1:{control_port}', 'metrics': f'http://127.0.0.1:{control_port}/metrics'}
                if endpoint != expected or any(saved.get(key) != value for key, value in expected.items()):
                    raise ExecutionError('Engine endpoints differ from the assigned identity and listeners')
                host['members'].add(node)
                self.instances[node] = host_name
                self.owner.urls[node] = endpoint['control']
                self.owner.topology.append(saved)
                self.owner.history.append({**saved, 'started_at': timestamp()})
                self.owner.publish_discovery()
                self.owner.records.emit(event='instance_started', node=node, host=host_name, pid=host['child'].pid, rpc=endpoint['rpc'])
                if self.owner.prometheus:
                    self.owner.prometheus.discover(node, directory)
                result.append(endpoint['rpc'])
            if self.owner.prometheus and self.owner.prometheus.baselined:
                await self.owner.prometheus.baseline([item['id'] for item in self.owner.topology if item['rpc'] in result])
            return result

    async def remove(self, role, targets, count):
        async with self.lock:
            candidates = [item for item in self.owner.topology if item['role'] == role and item['id'] in self.instances]
            known = {item['rpc']: item for item in candidates}
            if targets is not None:
                if not targets or len(set(targets)) != len(targets) or set(targets) - known.keys():
                    raise ExecutionError('Engine targets must be nonempty, unique, active addresses of the selected role')
                candidates = [known[address] for address in targets]
            if not candidates or count is not None and not 1 <= count <= len(candidates):
                raise ExecutionError('No Engine candidates or scale_in count exceeds the selected set')
            selected = candidates if count is None else candidates[:count]
            # Selection is fixed once, before any host-local stop is sent.
            self.owner.records.emit(event='instances_selected', operation='scale_in', targets=[x['rpc'] for x in selected])
            if self.owner.prometheus:
                await self.owner.prometheus.final_scrape([item['id'] for item in selected])
            for item in selected:
                node = item['id']
                host_name = self.instances[node]
                host = self.hosts[host_name]
                stopped = await self.call(host, 'stop', {'id': node})
                if stopped.get('id') != node or stopped.get('rpc') != item['rpc']:
                    raise ExecutionError('Host stopped an unexpected Engine')
                host['members'].remove(node)
                del self.instances[node]
                del self.owner.urls[node]
                self.owner.topology.remove(item)
                for historical in self.owner.history:
                    if historical['id'] == node:
                        historical['stopped_at'] = timestamp()
                self.owner.publish_discovery()
                if self.owner.prometheus:
                    self.owner.prometheus.retire(node)
                self.owner.records.emit(event='instance_stopped', node=node, host=host_name, rpc=item['rpc'])
                if self.owner.launch['engine_mode'] == 'isolated':
                    errors = await self.owner.stop_expected(host_name, host['child'])
                    del self.hosts[host_name]
                    if errors:
                        raise ExecutionError('; '.join(errors))
            return [item['rpc'] for item in selected]

    async def monitor(self):
        try:
            while not self.owner.closing:
                await asyncio.sleep(1)
                async with self.lock:
                    for name, host in self.hosts.items():
                        values = await self.call(host, 'status', {})
                        expected = {item['id']: item['rpc'] for item in self.owner.topology if item['id'] in host['members']}
                        if len(values) != len(expected) or {item['id']: item['rpc'] for item in values} != expected:
                            raise ExecutionError(f'Engine membership differs in {name}')
        except Exception as error:
            self.owner.fail(f'Engine host health failed: {error}')
