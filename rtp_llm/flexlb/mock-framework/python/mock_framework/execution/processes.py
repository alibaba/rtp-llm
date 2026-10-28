import asyncio
import copy
import errno
import json
import os
from pathlib import Path
import signal
import secrets
import socket
import uuid

from ..config.values import duration, fields, text
from .records import write_json, timestamp
from .transport import ExecutionError
from .monitoring import Prometheus, prepare_prometheus
from .master import master_environment, refresh_discovery
from .engines import EngineHosts, ROLES


def prepare_launch(raw, case, base):
    """Launch recipes are developer bindings, separate from case business configuration."""
    fields(raw, {'components', 'prometheus', 'zookeeper', 'engine_mode'}, {'components'}, where='launch')
    if not isinstance(raw['components'], dict):
        raise ExecutionError('launch components must be a mapping')
    if 'zookeeper' in raw['components']:
        raise ExecutionError('ZooKeeper belongs in the launch.zookeeper recipe, not components')
    components = {}
    recipes = dict(raw['components'])
    if 'zookeeper' in raw:
        recipes['zookeeper'] = raw['zookeeper']
    for role, recipe in recipes.items():
        if role != 'zookeeper' and role not in case.environment:
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
        deployment = case.environment[role]['deployment'] if role != 'zookeeper' else {}
        if deployment.get('resources') or deployment.get('heap'):
            raise ExecutionError('This local launcher does not yet enforce resources/heap; refusing to ignore them')
        components[role] = {**recipe, 'command': command, 'env': env, 'cwd': str(cwd.resolve()),
                            'startup_timeout': startup}
    zookeeper = components.pop('zookeeper', None)
    native_master = 'MODEL_SERVICE_CONFIG' in case.environment.get('master', {}).get('config', {})
    if native_master != (zookeeper is not None):
        raise ExecutionError('Native Master requires a zookeeper launch recipe; other components do not use it')
    missing = set(case.environment) - set(components)
    if missing:
        raise ExecutionError(f'Missing component launch recipes: {sorted(missing)}')
    mode = raw.get('engine_mode', 'shared')
    if mode not in ('shared', 'isolated'):
        raise ExecutionError('engine_mode must be shared or isolated')
    engine_recipes = [recipe for role, recipe in components.items() if role in ROLES]
    if mode == 'shared' and engine_recipes and any(recipe != engine_recipes[0] for recipe in engine_recipes[1:]):
        raise ExecutionError('Shared Engine JVM requires identical command, cwd, env and startup_timeout for all Engine roles; put instance settings in config')
    result = {'components': components, 'engine_mode': mode}
    if zookeeper is not None:
        result['zookeeper'] = zookeeper
    if 'prometheus' in raw:
        result['prometheus'] = prepare_prometheus(raw['prometheus'], base)
        if any('prometheus' in value.get('config', {}) for value in case.environment.values()):
            raise ExecutionError('Managed Prometheus cannot be combined with component prometheus configuration')
    return result


class Processes:
    def __init__(self, case, launch, records):
        self.case, self.launch, self.records = case, launch, records
        self.children, self.monitors, self.urls = [], [], {}
        self.failure = asyncio.get_running_loop().create_future()
        self.closing = False
        self.topology, self.history = [], []
        self.expected_stops = set()
        self.engines = EngineHosts(self)
        self.health_monitor = None
        self.client_pending = False
        self.client_stop_timeout = 30
        self.prometheus = Prometheus(self, records, launch['prometheus']) if 'prometheus' in launch else None

    async def _monitor(self, node, child):
        code = await child.wait()
        if node not in self.expected_stops:
            affected = sorted(self.engines.hosts.get(node, {}).get('members', ()))
            self.fail(f'Component {node} exited unexpectedly with code {code}; affected instances: {affected}')

    def fail(self, message):
        if not self.closing and not self.failure.done():
            self.failure.set_result(message)

    def publish_discovery(self):
        write_json(self.records.directory / 'topology.json', self.topology)
        write_json(self.records.directory / 'topology-history.json', self.history)
        for master in self.history:
            if master['role'] == 'master':
                refresh_discovery(self.case.environment['master']['config'], self.topology,
                                  self.records.directory / 'components' / master['id'])

    async def stop_expected(self, node, child):
        if child.returncode is not None:
            raise ExecutionError(f'{node} already exited before intentional stop')
        self.expected_stops.add(node)
        errors = await self._stop([(node, child)], 3)
        if child.returncode is not None:
            self.children.remove((node, child))
        return errors

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

    async def spawn(self, node, command, cwd, env, directory):
        with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                *command, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True))
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
        return child

    async def start(self):
        if self.prometheus:
            await self.guard(self.prometheus.start(), self.prometheus.startup_timeout)
        master_env = {}
        if 'zookeeper' in self.launch:
            recipe = self.launch['zookeeper']
            directory = self.records.directory / 'zookeeper'
            directory.mkdir()
            ready = directory / 'ready.json'
            await self.spawn('zookeeper', recipe['command'], recipe['cwd'],
                             {**os.environ, **recipe['env'], 'MOCK_OUTPUT_DIR': str(directory),
                              'MOCK_READY_FILE': str(ready)}, directory)
            async def zk_ready():
                while not ready.exists():
                    await asyncio.sleep(.02)
                value = json.loads(ready.read_text())
                host, port = value['connect'].split(':')
                if host != '127.0.0.1' or not 0 < int(port) < 65536:
                    raise ExecutionError('Invalid ZooKeeper readiness address')
                return value['connect']
            connect = await self.guard(zk_ready(), float(duration(recipe['startup_timeout'])))
            master_env = {'HIPPO_ROLE': 'mock-' + uuid.uuid4().hex,
                          'FLEXLB_SYNC_CONSISTENCY_CONFIG': json.dumps({'needConsistency': True,
                              'zookeeperConfig': {'zkHost': connect, 'zkTimeoutMs': 6000}})}
            self.records.emit(event='process_ready', node='zookeeper', connect=connect)
        topology = self.topology
        order = {'prefill': 0, 'decode': 0, 'vit': 0, 'master': 1, 'client': 2}
        for role, recipe in sorted(self.launch['components'].items(), key=lambda item: order[item[0]]):
            if role in ROLES:
                await self.guard(self.engines.create(role, self.case.environment[role]['deployment']['replicas']),
                                 float(duration(recipe['startup_timeout'])) * self.case.environment[role]['deployment']['replicas'])
                continue
            for index in range(self.case.environment[role]['deployment']['replicas']):
                node = f'{role}-{index}'
                directory = self.records.directory / 'components' / node
                directory.mkdir(parents=True)
                config = directory / 'config.json'
                ready = directory / 'ready.json'
                component = copy.deepcopy(self.case.environment[role])
                write_json(config, {'component_role': role, 'component': component,
                                    'environment': self.case.environment, 'index': index, 'id': node})
                native_env = master_environment(component['config'], topology, directory) if role == 'master' else {}
                if role == 'master' and master_env:
                    if index == 0:
                        master_env['MOCK_MASTER_PORT'] = str(master_port(self.case.environment[role]['deployment']['replicas']))
                        write_json(self.records.directory / 'zookeeper/master-environment.json', master_env)
                    native_env.update(master_env, FLEXLB_ADVERTISED_IP=f'127.0.{index // 254}.{index % 254 + 1}')
                    write_json(directory / 'native-environment.json', native_env)
                topology_file = self.records.directory / 'topology.json'
                write_json(topology_file, topology)
                env = {**os.environ, **recipe['env'], **native_env, 'MOCK_TOPOLOGY_FILE': str(topology_file), 'MOCK_TOPOLOGY_HISTORY_FILE': str(self.records.directory / 'topology-history.json'), 'MOCK_NODE_ID': node, 'MOCK_COMPONENT': role,
                       'MOCK_CONFIG_FILE': str(config), 'MOCK_READY_FILE': str(ready), 'MOCK_OUTPUT_DIR': str(directory)}
                await self.spawn(node, recipe['command'], recipe['cwd'], env, directory)
                async def await_ready():
                    while not ready.exists():
                        await asyncio.sleep(.02)
                    value = json.loads(ready.read_text(encoding='utf-8'))
                    if set(value) != {'url'} or not isinstance(value['url'], str):
                        raise ExecutionError(f'Invalid readiness file for {node}')
                    self.urls[node] = value['url']
                await self.guard(await_ready(), float(duration(recipe['startup_timeout'])))
                self.records.emit(event='process_ready', node=node, url=self.urls[node])
                endpoints = directory / 'endpoints.json'
                if endpoints.is_file():
                    value = json.loads(endpoints.read_text(encoding='utf-8'))
                    topology.append({**value, 'id': node, 'role': role})
                    self.history.append({**value, 'id': node, 'role': role, 'started_at': timestamp()})
                    self.publish_discovery()
                if self.prometheus:
                    self.prometheus.discover(node, directory)
        if self.engines.hosts:
            self.health_monitor = asyncio.create_task(self.engines.monitor())
        if self.prometheus:
            await self.guard(self.prometheus.baseline(), self.prometheus.startup_timeout)

    async def _stop(self, children, grace):
        errors = []
        # Signal only this run's owned groups, including grandchildren if the leader exited.
        for node, child in reversed(children):
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{node}: {error}')
        deadline = asyncio.get_running_loop().time() + grace
        while any(child.returncode is None for _, child in children) and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(.02)
        for node, child in reversed(children):
            killed = False
            try:
                os.killpg(child.pid, signal.SIGKILL)
                killed = child.returncode is None
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f'{node}: {error}')
            try:
                await asyncio.wait_for(child.wait(), timeout=3)
            except TimeoutError:
                errors.append(f'{node}: process did not exit after SIGKILL')
            if killed:
                errors.append(f'{node}: did not stop within the grace period; SIGKILL was required')
            if node in self.expected_stops and child.returncode not in (0, -signal.SIGTERM, 128 + signal.SIGTERM):
                errors.append(f'{node}: unexpected exit status during stop: {child.returncode}')
            if node == 'prometheus' and (killed or child.returncode != 0):
                errors.append('prometheus: did not stop gracefully; retained data may be incomplete')
            self.records.emit(event='process_stopped', node=node, code=child.returncode)
        return errors

    async def close(self):
        self.closing = True
        if self.health_monitor:
            self.health_monitor.cancel()
            await asyncio.gather(self.health_monitor, return_exceptions=True)
        errors = []
        if self.client_pending:
            from ..interfaces.client import collect
            try:
                async with asyncio.timeout(self.client_stop_timeout):
                    await collect(self, self.records, timestamp(), self.client_stop_timeout)
            except Exception as error:
                errors.append(f'client final stop: {type(error).__name__}: {error}')
        if self.prometheus:
            try:
                await self.prometheus.final_scrape()
            except Exception as error:
                errors.append(f'prometheus final scrape: {type(error).__name__}: {error}')
        errors.extend(await self._stop([(node, child) for node, child in self.children if node not in ('prometheus', 'zookeeper')], 3))
        errors.extend(await self._stop([(node, child) for node, child in self.children if node == 'zookeeper'], 5))
        errors.extend(await self._stop([(node, child) for node, child in self.children if node == 'prometheus'], 5))
        if all(child.returncode is not None for _, child in self.children):
            self.topology.clear()
            for item in self.history:
                item.setdefault('stopped_at', timestamp())
            try:
                self.publish_discovery()
            except Exception as error:
                errors.append(f'discovery cleanup: {error}')
        if self.prometheus:
            self.prometheus.finish(errors)
        for monitor in self.monitors:
            if not monitor.done():
                monitor.cancel()
        await asyncio.gather(*self.monitors, return_exceptions=True)
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
