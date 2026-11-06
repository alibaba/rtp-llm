"""Assemble run-scoped services and preserve their explicit startup/cleanup order."""
from .roles.client import ClientManager
from .roles.engine import EngineRole
from .roles.master import MasterRole
from .roles.prometheus import Prometheus
from .roles.zookeeper import ZooKeeperRole
from .processes import ProcessSupervisor
import copy

from .run_trace import write_json
from . import ExecutionError


class EndpointRegistry:
    def __init__(self, directory):
        self.directory = directory
        self._active = {}
        self._publishers = []

    def active(self, role=None):
        return copy.deepcopy([node for node in self._active.values() if role is None or node['role'] == role])

    def add_publisher(self, publish):
        self._publishers.append(publish)

    def publish(self):
        snapshot = self.active()
        write_json(self.directory / 'topology.json', snapshot)
        for publish in self._publishers:
            publish(copy.deepcopy(snapshot))

    def register(self, name, role, control, endpoint=None):
        if name in self._active:
            raise ExecutionError(f'Instance identity has already been used: {name}')
        if endpoint is not None:
            if endpoint.get('id', name) != name or endpoint.get('role', role) != role or endpoint.get('control') != control:
                raise ExecutionError('Instance endpoints differ from assigned identity')
            endpoint = {**copy.deepcopy(endpoint), 'id': name, 'role': role}
            if not isinstance(endpoint.get('rpc'), str) or not endpoint['rpc'] or any(node['rpc'] == endpoint['rpc'] for node in self._active.values()):
                raise ExecutionError('Instance RPC address is missing or already active')
            self._active[name] = endpoint
        self.publish()

    def retire(self, name):
        if name not in self._active:
            raise ExecutionError(f'Unknown active instance: {name}')
        self._active.pop(name, None)
        self.publish()

    def clear(self):
        self._active.clear()
        self.publish()


class RunEnvironment:
    def __init__(self, case, launch, records):
        self.case, self.records = case, records
        self.supervisor = ProcessSupervisor(records)
        self.zookeeper = ZooKeeperRole(launch.get('zookeeper'), self.supervisor, records)
        self.topology = EndpointRegistry(records.directory)
        self.monitoring = Prometheus(self.supervisor, records, launch['prometheus']) if 'prometheus' in launch else None
        self.masters = MasterRole(component=case.environment.get('master'), recipe=launch['components'].get('master'),
            environment=case.environment, election=launch.get('zookeeper'), supervisor=self.supervisor,
            topology=self.topology, records=records, monitoring=self.monitoring)
        self.topology.add_publisher(self.masters.publish_discovery)
        self.engines = EngineRole(supervisor=self.supervisor, topology=self.topology, records=records,
            environment=case.environment, recipes=launch['components'], monitoring=self.monitoring)
        self.clients = ClientManager(environment=case.environment, component=case.environment.get('client'), recipe=launch['components'].get('client'),
            supervisor=self.supervisor, topology=self.topology, records=records, monitoring=self.monitoring)

    async def start(self):
        self.topology.publish()
        if self.monitoring:
            await self.supervisor.guard(self.monitoring.start(), self.monitoring.startup_timeout)
        connect = await self.zookeeper.start()
        if connect is not None:
            self.masters.configure_election(connect)
            # Masters start after Engines: exclude their selected but not yet bound service ports.
            port = int(self.masters.environment['MOCK_MASTER_PORT'])
            self.engines.used_ports.update((port, port + 2))
        await self.engines.start()
        await self.masters.start()
        await self.clients.start_nodes()
        if self.monitoring:
            await self.supervisor.guard(self.monitoring.baseline(), self.monitoring.startup_timeout)

    async def close(self):
        self.supervisor.closing = True
        errors = []
        async def cleanup(label, operation):
            try:
                result = await operation()
                if result:
                    errors.extend(result)
            except Exception as error:
                errors.append(f'{label}: {type(error).__name__}: {error}')
        if self.supervisor.failure.done():
            errors.append(self.supervisor.failure.result())
        # Keep producers and observers alive through Client drain and the final scrape.
        await cleanup('client final stop', self.clients.stop_playback)
        if self.monitoring:
            await cleanup('prometheus final scrape', self.monitoring.final_scrape)
        await cleanup('client processes', self.clients.close)
        await cleanup('master processes', self.masters.close)
        await cleanup('engine processes', self.engines.close)
        await cleanup('zookeeper', self.zookeeper.close)
        if self.monitoring:
            await cleanup('prometheus', self.monitoring.close)
        await cleanup('remaining processes', self.supervisor.close)
        if not self.supervisor.children:
            try:
                self.topology.clear()
            except Exception as error:
                errors.append(f'discovery cleanup: {error}')
        if self.monitoring:
            try:
                self.monitoring.finish(errors)
            except Exception as error:
                errors.append(f'collection metadata: {error}')
        return errors
