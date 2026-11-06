"""Native Master configuration binding and per-Master state reads."""
import asyncio
import copy
from dataclasses import dataclass
import errno
import json
import os
import secrets
import socket
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from ...config.validation import ConfigError, text

from ..processes import launch_component
from ..step_trace import write_json
from .. import ExecutionError
from ..transport import post


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f'Duplicate native JSON key: {key}')
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def validate_master_config(config):
    if 'FLEXLB_CONFIG' not in config:
        return
    encoded = text(config['FLEXLB_CONFIG'], 'FLEXLB_CONFIG')
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_json_object,
                           parse_constant=_reject_json_constant)
    except (ValueError, TypeError) as error:
        raise ConfigError(f'Invalid FLEXLB_CONFIG JSON: {error}') from error
    if not isinstance(value, dict):
        raise ConfigError('FLEXLB_CONFIG must contain a JSON object')


def validate_master_launch(config, launch_env, *, local_election):
    declared = {key for key in config if key.isupper()}
    duplicate = declared & launch_env.keys()
    if duplicate:
        key = min(duplicate)
        raise ConfigError(f'{key} is declared in both environment.master.config and launch.components.master.env')
    assigned = {'MOCK_TOPOLOGY_FILE', 'MOCK_NODE_ID', 'MOCK_COMPONENT', 'MOCK_CONFIG_FILE',
                'MOCK_READY_FILE', 'MOCK_OUTPUT_DIR'}
    if local_election:
        assigned.update({'HIPPO_ROLE', 'FLEXLB_SYNC_CONSISTENCY_CONFIG',
                         'FLEXLB_ADVERTISED_IP', 'MOCK_MASTER_PORT'})
    for source, keys in [('environment.master.config', declared), ('launch.components.master.env', launch_env.keys())]:
        conflict = assigned & keys
        if conflict:
            raise ConfigError(f'{source}.{min(conflict)} is assigned by the local runner')


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


@dataclass
class _MasterInstance:
    index: int
    attempt: int
    process: str
    directory: Path
    name: str


class MasterRole:
    def __init__(self, *, component, recipe, election, supervisor, topology, records, monitoring, environment=None):
        self.component, self.recipe, self.election = component, recipe, election
        self.supervisor, self.topology = supervisor, topology
        self._configuration = environment
        self.records, self.monitoring = records, monitoring
        self.environment, self._owned = {}, []
        self._next = 0
        self._instances = {}
        self._lock = asyncio.Lock()

    def configure_election(self, connect):
        directory = self.records.directory / 'zookeeper'
        self.environment = {'HIPPO_ROLE': 'mock-' + uuid.uuid4().hex,
            'FLEXLB_SYNC_CONSISTENCY_CONFIG': json.dumps({'needConsistency': True,
                'zookeeperConfig': {'zkHost': connect, 'zkTimeoutMs': 6000}}),
            'MOCK_MASTER_PORT': str(master_port(self.component['deployment']['replicas']))}
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
                    instance = self._instances.get(node['rpc'])
                    directory = instance.directory if instance else self.records.directory / 'components' / node['id']
                    refresh_discovery(self.component['config'], snapshot, directory)

    async def _launch(self, index, attempt=0):
        name = f'master-{index}'
        process = name if attempt == 0 else f'{name}-restart-{attempt}'
        directory = self.records.directory / 'components' / name
        if attempt:
            directory = directory / f'restart-{attempt}'
        self._owned.append(process)
        if attempt:
            address = next(address for address, item in self._instances.items() if item.index == index)
            self._instances[address].directory = directory
        control, endpoint = await launch_component(
            self.supervisor, self.records, self._configuration, 'master', index, self.recipe,
            configure=lambda directory: self._environment(index, directory), attempt=attempt)
        if endpoint is None and self.environment:
            raise ExecutionError('Native Master did not publish endpoints')
        if endpoint is not None:
            self._validate(name, index, endpoint, control)
        self.topology.register(name, 'master', control, endpoint)
        if self.monitoring:
            self.monitoring.discover(name, directory)
            if self.monitoring.baselined:
                await self.monitoring.baseline([name])
        return endpoint, _MasterInstance(index, attempt, process, directory, name)

    async def start(self, count=None):
        if count is None:
            if self.component is None:
                return []
            count = self.component['deployment']['replicas']
        async with self._lock:
            if self.recipe is None or not 1 <= count <= 1024:
                raise ExecutionError('master/start requires configured Masters and count in 1..1024')
            result = []
            for _ in range(count):
                index = self._next
                self._next += 1
                endpoint, instance = await self._launch(index)
                if endpoint is not None:
                    self._instances[endpoint['rpc']] = instance
                    result.append(endpoint['rpc'])
            return result

    def _read_targets(self, targets):
        available = {item['rpc']: item for item in self.topology.active('master')}
        addresses = sorted(available) if targets is None else targets
        if not addresses or set(addresses) - available.keys():
            raise ExecutionError('Master targets are empty or unknown')
        return [available[address] for address in addresses]

    async def _read_state(self, master, operation, args, query_time, remaining):
        return await post(master['control'], {'interface': operation, 'args': args,
            'context': {'run': 'foreground', 'query_time': query_time, 'observation': None,
                        'timeout_ms': max(1, int(remaining() * 1000))}})

    async def get(self, targets, is_leader, *, query_time, remaining):
        masters = self._read_targets(targets)
        if is_leader is None:
            return [master['rpc'] for master in masters]
        result = []
        for master in masters:
            value = await self._read_state(master, 'master/leadership/get', {}, query_time, remaining)
            own = value['instances']
            if own not in ([], [{'id': master['id']}]):
                raise ExecutionError('Master returned an invalid local leadership observation')
            if bool(own) == is_leader:
                result.append(master['rpc'])
        return result

    async def get_engines(self, targets, role, *, query_time, remaining):
        result = {}
        for master in self._read_targets(targets):
            value = await self._read_state(master, 'master/engine/get', {'role': role}, query_time, remaining)
            if set(value['instances']) != {master['id']}:
                raise ExecutionError('Master returned another instance observation')
            addresses = value['instances'][master['id']]
            if not isinstance(addresses, list) or any(not isinstance(address, str) or not address for address in addresses):
                raise ExecutionError('Master reported invalid Engine addresses')
            if len(set(addresses)) != len(addresses):
                raise ExecutionError('Master reported duplicate Engines')
            result[master['rpc']] = sorted(addresses)
        return result

    def _selected(self, targets, *, active):
        available = {node['rpc'] for node in self.topology.active('master')}
        candidates = available if active else self._instances.keys() - available
        addresses = sorted(candidates) if targets is None else targets
        if not addresses or len(set(addresses)) != len(addresses) or set(addresses) - candidates:
            state = 'active' if active else 'stopped'
            raise ExecutionError(f'Master targets must be nonempty, unique, {state} addresses')
        return list(addresses)

    async def stop(self, targets, mode='terminate'):
        async with self._lock:
            if mode not in ('terminate', 'kill'):
                raise ExecutionError('Unknown Master stop mode')
            addresses = self._selected(targets, active=True)
            selected = [self._instances[address] for address in addresses]
            if any(self.supervisor.child(item.process).returncode is not None for item in selected):
                raise ExecutionError('Selected Master is not running')
            self.records.emit(event='instances_selected', operation='master/stop', targets=addresses, mode=mode)
            if self.monitoring:
                await self.monitoring.final_scrape([item.name for item in selected])
            for item in selected:
                child = self.supervisor.child(item.process)
                errors = await self.supervisor.stop_intentionally(item.process, mode=mode, grace=self._stop_grace())
                if child.returncode is not None:
                    (item.directory / 'ready.json').unlink(missing_ok=True)
                    self.topology.retire(item.name)
                    self.records.emit(event='instance_stopped', node=item.name, role='master',
                                      instance_id=item.name, generation=item.attempt, process=item.process)
                    if self.monitoring:
                        self.monitoring.retire(item.name)
                if errors:
                    raise ExecutionError('; '.join(errors))
            return addresses

    async def restart(self, targets):
        async with self._lock:
            addresses = self._selected(targets, active=False)
            for address in addresses:
                item = self._instances[address]
                item.attempt += 1
                endpoint, replacement = await self._launch(item.index, item.attempt)
                if endpoint is None or endpoint['rpc'] != address:
                    raise ExecutionError('Restarted Master changed its service address')
                self._instances[address] = replacement
            return addresses

    def _stop_grace(self):
        # Native shutdown includes up to 30s of leadership transfer, quiet time and resource teardown.
        return 60 if self.election is not None else 3

    async def close(self):
        errors = []
        # Keep peers and ZooKeeper alive until each Master has completed its shutdown.
        for process in reversed(self._owned):
            errors.extend(await self.supervisor.stop([process], self._stop_grace()))
        return errors


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
                # Match the native listener: TIME_WAIT must not prevent an owned address from restarting.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, port + offset))
    except OSError as error:
        detail = 'enable loopback aliases or run on Linux' if error.errno == errno.EADDRNOTAVAIL else str(error)
        raise ExecutionError(f'Cannot bind Master {host}:{port}: {detail}') from error
