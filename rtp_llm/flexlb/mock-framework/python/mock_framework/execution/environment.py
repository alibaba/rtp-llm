"""Assemble run-scoped services and preserve their explicit startup/cleanup order."""
from .clients import ClientGroup
from .components import ComponentLauncher
from .engines import EngineHosts
from .master import MasterGroup
from .monitoring import Prometheus
from .processes import ProcessSupervisor
from .topology import Topology


class RunEnvironment:
    def __init__(self, case, launch, records):
        self.case, self.records = case, records
        self.supervisor = ProcessSupervisor(records)
        self.topology = Topology(records.directory)
        self.monitoring = Prometheus(self.supervisor, records, launch['prometheus']) if 'prometheus' in launch else None
        launcher = ComponentLauncher(self.supervisor, self.topology, records, case.environment)
        self.masters = MasterGroup(component=case.environment.get('master'), recipe=launch['components'].get('master'),
            election=launch.get('zookeeper'), launcher=launcher, supervisor=self.supervisor,
            topology=self.topology, records=records, monitoring=self.monitoring)
        self.topology.add_publisher(self.masters.publish_discovery)
        self.engines = EngineHosts(supervisor=self.supervisor, topology=self.topology, records=records,
            environment=case.environment, recipes=launch['components'], mode=launch['engine_mode'], monitoring=self.monitoring)
        self.clients = ClientGroup(component=case.environment.get('client'), recipe=launch['components'].get('client'),
            launcher=launcher, supervisor=self.supervisor, topology=self.topology, records=records, monitoring=self.monitoring)

    async def start(self):
        self.topology.publish()
        if self.monitoring:
            await self.supervisor.guard(self.monitoring.start(), self.monitoring.startup_timeout)
        await self.masters.prepare()
        await self.engines.start()
        if self.masters.component is not None:
            await self.masters.start(self.masters.component['deployment']['replicas'])
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
        await cleanup('engine health monitor', self.engines.stop_monitor)
        if self.supervisor.failure.done():
            errors.append(self.supervisor.failure.result())
        # Keep producers and observers alive through Client drain and the final scrape.
        await cleanup('client final stop', self.clients.drain)
        if self.monitoring:
            await cleanup('prometheus final scrape', self.monitoring.final_scrape)
        await cleanup('client processes', self.clients.close)
        await cleanup('master processes', self.masters.close)
        await cleanup('engine processes', self.engines.close)
        await cleanup('zookeeper', self.masters.close_election)
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
