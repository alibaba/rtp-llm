"""Native Master configuration binding and per-Master state reads."""
import asyncio
import copy
import errno
import json
import os
import secrets
import socket
import uuid
from urllib.parse import urlsplit
from ..config.values import duration

from .records import write_json
from .transport import ExecutionError


def _environment(config, topology, directory, *, require_engines):
    env = {}
    for key, value in config.items():
        if key.isupper():
            if not isinstance(value, str):
                raise ExecutionError(f'Master native setting {key} must be an environment string')
            env[key] = value
    if 'MODEL_SERVICE_CONFIG' not in env:
        return env  # Generic transport probes do not launch the native Master.
    try:
        route = json.loads(env['MODEL_SERVICE_CONFIG'])
        groups = route['role_endpoints']
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError('local Master currently requires exactly one endpoint group')
        route = copy.deepcopy(route)
        hosts = {}
        configured = {node['role'] for node in topology}
        for role in ('prefill', 'decode', 'vit'):
            endpoint = route['role_endpoints'][0].get(role + '_endpoint')
            if endpoint is None:
                if role in configured:
                    raise ValueError(f'Missing native {role}_endpoint')
                continue
            if endpoint.get('protocol') != 'grpc' or not isinstance(endpoint.get('address'), str) or not endpoint['address']:
                raise ValueError('local Engine endpoints require grpc protocol and a discovery address')
            domain = endpoint['address']
            if domain in hosts:
                raise ValueError('Each Engine role needs a distinct discovery address')
            addresses = []
            for node in topology:
                if node['role'] == role:
                    host, port = node['rpc'].rsplit(':', 1)
                    # Native WorkerAddressService translates grpc discovery ports to its HTTP identity.
                    port = int(port)
                    if port <= 0:
                        raise ValueError('Invalid Engine RPC port')
                    addresses.append(f'{host}:{port}')
            if require_engines and not addresses:
                raise ValueError(f'No running Engines for native {role}_endpoint')
            hosts[domain] = addresses
        if not hosts or route['role_endpoints'][0].get('pd_fusion_endpoint') is not None:
            raise ValueError('local Master supports separate P/D/V endpoints only')
        discovery = directory / 'discovery.json'
        write_json(discovery, hosts)
        route.pop('hosts', None)
        route['discovery_file'] = str(discovery)
        env['MODEL_SERVICE_CONFIG'] = json.dumps(route, ensure_ascii=False)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ExecutionError(f'Cannot bind local MODEL_SERVICE_CONFIG: {error}') from error
    return env


def master_environment(config, topology, directory, *, require_engines=True):
    env = _environment(config, topology, directory, require_engines=require_engines)
    write_json(directory / 'native-environment.json', env)
    return env


def refresh_discovery(config, topology, directory):
    if 'MODEL_SERVICE_CONFIG' in config:
        # The consumer owns reload timing; only publish the current addresses here.
        _environment(config, topology, directory, require_engines=False)


class MasterGroup:
    def __init__(self, *, component, recipe, election, launcher, supervisor, topology, records, monitoring):
        self.component, self.recipe, self.election = component, recipe, election
        self.launcher, self.supervisor, self.topology = launcher, supervisor, topology
        self.records, self.monitoring = records, monitoring
        self.environment, self._owned = {}, []
        self._next = 0
        self._lock = asyncio.Lock()

    async def prepare(self):
        if self.election is None:
            return
        recipe = self.election
        directory = self.records.directory / 'zookeeper'
        directory.mkdir()
        ready = directory / 'ready.json'
        await self.supervisor.spawn('zookeeper', recipe['command'], recipe['cwd'],
            {**os.environ, **recipe['env'], 'MOCK_OUTPUT_DIR': str(directory), 'MOCK_READY_FILE': str(ready)}, directory)
        async def await_ready():
            while not ready.exists():
                await asyncio.sleep(.02)
            value = json.loads(ready.read_text())
            host, port = value['connect'].split(':')
            if host != '127.0.0.1' or not 0 < int(port) < 65536:
                raise ExecutionError('Invalid ZooKeeper readiness address')
            return value['connect']
        connect = await self.supervisor.guard(await_ready(), float(duration(recipe['startup_timeout'])))
        self.environment = {'HIPPO_ROLE': 'mock-' + uuid.uuid4().hex,
            'FLEXLB_SYNC_CONSISTENCY_CONFIG': json.dumps({'needConsistency': True,
                'zookeeperConfig': {'zkHost': connect, 'zkTimeoutMs': 6000}}),
            'MOCK_MASTER_PORT': str(master_port(self.component['deployment']['replicas']))}
        self.records.emit(event='process_ready', node='zookeeper', connect=connect)
        write_json(directory / 'master-environment.json', self.environment)

    def _environment(self, index, directory):
        env = master_environment(self.component['config'], self.topology.active(), directory,
            require_engines=index < self.component['deployment']['replicas'])
        if self.environment:
            host = master_host(index)
            check_master_ports(host, int(self.environment['MOCK_MASTER_PORT']))
            env.update(self.environment, FLEXLB_ADVERTISED_IP=host)
            write_json(directory / 'native-environment.json', env)
        return env

    def _validate(self, name, index, endpoint, control):
        if not self.environment:
            return
        host, port = master_host(index), int(self.environment['MOCK_MASTER_PORT'])
        address = urlsplit(control)
        if address.scheme != 'http' or address.hostname != host or not address.port or address.port in (port, port + 2):
            raise ExecutionError('Master control address differs from assigned host')
        expected = {'id': name, 'role': 'master', 'http': f'http://{host}:{port}',
                    'rpc': f'{host}:{port + 2}', 'control': control, 'metrics': f'http://{host}:{port}/prometheus'}
        if endpoint != expected:
            raise ExecutionError('Master endpoints differ from assigned identity and listeners')

    def publish_discovery(self, snapshot):
        if self.component is not None:
            for node in snapshot:
                if node['role'] == 'master':
                    refresh_discovery(self.component['config'], snapshot, self.records.directory / 'components' / node['id'])

    async def start(self, count):
        async with self._lock:
            if self.recipe is None or not 1 <= count <= 1024:
                raise ExecutionError('master/start requires configured Masters and count in 1..1024')
            result, nodes = [], []
            for _ in range(count):
                index = self._next
                self._next += 1
                name = f'master-{index}'
                self._owned.append(name)
                endpoint = await self.launcher.start('master', index, self.recipe,
                    configure=lambda directory: self._environment(index, directory),
                    validate=lambda value, control: self._validate(name, index, value, control),
                    require_endpoints=bool(self.environment))
                if self.monitoring:
                    self.monitoring.discover(name, self.records.directory / 'components' / name)
                if endpoint is not None:
                    result.append(endpoint['rpc']); nodes.append(name)
            if self.monitoring and self.monitoring.baselined and nodes:
                await self.monitoring.baseline(nodes)
            return result

    async def stop(self, targets):
        async with self._lock:
            available = {node['rpc']: node for node in self.topology.active('master')}
            addresses = sorted(available) if targets is None else targets
            if not addresses or len(set(addresses)) != len(addresses) or set(addresses) - available.keys():
                raise ExecutionError('Master targets must be nonempty, unique, active addresses')
            selected = [available[address] for address in addresses]
            if any(self.supervisor.child(node['id']).returncode is not None for node in selected):
                raise ExecutionError('Selected Master is not running')
            self.records.emit(event='instances_selected', operation='master/stop', targets=addresses)
            if self.monitoring:
                await self.monitoring.final_scrape([node['id'] for node in selected])
            for node in selected:
                name = node['id']
                child = self.supervisor.child(name)
                errors = await self.supervisor.stop_expected(name)
                if child.returncode is not None:
                    (self.records.directory / 'components' / name / 'ready.json').unlink(missing_ok=True)
                    self.topology.retire(name)
                    if self.monitoring:
                        self.monitoring.retire(name)
                if errors:
                    raise ExecutionError('; '.join(errors))
            return list(addresses)

    async def close(self):
        return await self.supervisor.stop(self._owned, 3)

    async def close_election(self):
        return await self.supervisor.stop(['zookeeper'], 5)


def master_port(count):
    """All Masters use distinct loopback IPv4 addresses and the same native ports."""
    if not 1 <= count <= 254 * 256:
        raise ExecutionError('Too many local Masters for the loopback address allocation')
    for attempt in range(32):
        sockets = []
        # Avoid client connections taking the selected port during JVM startup.
        port = 10000 + secrets.randbelow(20000)
        try:
            for index in range(count):
                host = f'127.0.{index // 254}.{index % 254 + 1}'
                for offset in (0, 2):
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind((host, port + offset))
            return port
        except OSError as error:
            if error.errno == errno.EADDRNOTAVAIL:
                raise ExecutionError(f'Cannot bind {host}: enable loopback aliases or run on Linux') from error
            if attempt == 31:
                raise ExecutionError('No available shared Master HTTP/gRPC port pair') from error
        finally:
            for sock in sockets:
                sock.close()


def master_host(index):
    if not 0 <= index < 254 * 256:
        raise ExecutionError('Local Master identity range exhausted')
    return f'127.0.{index // 254}.{index % 254 + 1}'


def check_master_ports(host, port):
    try:
        for offset in (0, 2):
            with socket.socket() as probe:
                probe.bind((host, port + offset))
    except OSError as error:
        detail = 'enable loopback aliases or run on Linux' if error.errno == errno.EADDRNOTAVAIL else str(error)
        raise ExecutionError(f'Cannot bind Master {host}:{port}: {detail}') from error
