"""Assemble run-scoped services and preserve their explicit startup/cleanup order."""

import copy

from . import ExecutionError
from .processes import ProcessSupervisor
from .roles import ENGINE_ROLES
from .roles.client import ClientRole
from .roles.engine import EngineRole
from .roles.master import MasterRole
from .roles.prometheus import PrometheusRole
from .roles.zookeeper import ZooKeeperRole
from .step_trace import write_json


class EndpointRegistry:
    def __init__(self, directory):
        self.directory = directory
        self._active = {}
        self._publish_discovery = None

    def active(self, role=None):
        return copy.deepcopy(
            [
                node
                for node in self._active.values()
                if role is None or node["role"] == role
            ]
        )

    def set_discovery_publisher(self, publish):
        self._publish_discovery = publish

    def publish(self, *, engine_changed=False):
        snapshot = self.active()
        write_json(self.directory / "topology.json", snapshot)
        if engine_changed and self._publish_discovery is not None:
            self._publish_discovery(snapshot)

    def register(self, instance_id, role, control, endpoint=None):
        if instance_id in self._active:
            raise ExecutionError(
                f"Instance identity has already been used: {instance_id}"
            )
        if endpoint is not None:
            if (
                endpoint.get("id", instance_id) != instance_id
                or endpoint.get("role", role) != role
                or endpoint.get("control") != control
            ):
                raise ExecutionError("Instance endpoints differ from assigned identity")
            endpoint = {**copy.deepcopy(endpoint), "id": instance_id, "role": role}
            if (
                not isinstance(endpoint.get("rpc"), str)
                or not endpoint["rpc"]
                or any(node["rpc"] == endpoint["rpc"] for node in self._active.values())
            ):
                raise ExecutionError(
                    "Instance RPC address is missing or already active"
                )
            self._active[instance_id] = endpoint
        self.publish(engine_changed=endpoint is not None and role in ENGINE_ROLES)

    def retire(self, instance_id):
        if instance_id not in self._active:
            raise ExecutionError(f"Unknown active instance: {instance_id}")
        endpoint = self._active.pop(instance_id)
        self.publish(engine_changed=endpoint["role"] in ENGINE_ROLES)

    def clear(self):
        self._active.clear()
        self.publish()


class RunEnvironment:
    def __init__(self, case, launch, records):
        self.case, self.records = case, records
        self.supervisor = ProcessSupervisor(records)
        self.zookeeper = ZooKeeperRole(
            launch.get("zookeeper"), self.supervisor, records
        )
        self.topology = EndpointRegistry(records.directory)
        self.monitoring = (
            PrometheusRole(self.supervisor, records, launch["prometheus"])
            if "prometheus" in launch
            else None
        )
        self.masters = MasterRole(
            role_config=case.environment.get("master"),
            launch_config=launch["components"].get("master"),
            election=launch.get("zookeeper"),
            supervisor=self.supervisor,
            topology=self.topology,
            records=records,
            monitoring=self.monitoring,
        )
        self.topology.set_discovery_publisher(self.masters.publish_discovery)
        self.engines = EngineRole(
            supervisor=self.supervisor,
            topology=self.topology,
            records=records,
            environment=case.environment,
            launch_by_role=launch["components"],
            monitoring=self.monitoring,
        )
        self.clients = ClientRole(
            role_config=case.environment.get("client"),
            launch_config=launch["components"].get("client"),
            supervisor=self.supervisor,
            topology=self.topology,
            records=records,
            monitoring=self.monitoring,
        )

    async def start(self):
        self.topology.publish()
        if self.monitoring:
            await self.supervisor.guard(
                self.monitoring.start(), self.monitoring.startup_timeout
            )
        connect = await self.zookeeper.start()
        if connect is not None:
            self.masters.configure_election(connect)
            # Masters start after Engines: exclude their selected but not yet bound service ports.
            port = int(self.masters.election_env["MOCK_MASTER_PORT"])
            self.engines.used_ports.update((port, port + 2))
        await self.engines.start()
        await self.masters.start()
        await self.clients.start()
        if self.monitoring:
            await self.supervisor.guard(
                self.monitoring.wait_for_baseline(), self.monitoring.startup_timeout
            )

    async def close(self):
        self.supervisor.closing = True
        errors = []

        async def cleanup(label, operation):
            try:
                result = await operation()
                if result:
                    errors.extend(result)
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")

        async def drain_clients():
            await self.clients.stop_playback()

        if self.supervisor.failure.done():
            errors.append(self.supervisor.failure.result())
        # Keep producers and observers alive through Client drain and the final scrape.
        await cleanup("client final stop", drain_clients)
        if self.monitoring:
            await cleanup(
                "prometheus final scrape", self.monitoring.wait_for_final_scrape
            )
        await cleanup("client processes", self.clients.close)
        await cleanup("master processes", self.masters.close)
        await cleanup("engine processes", self.engines.close)
        await cleanup("zookeeper", self.zookeeper.close)
        if self.monitoring:
            await cleanup("prometheus", self.monitoring.close)
        await cleanup("remaining processes", self.supervisor.close)
        if not self.supervisor.children:
            try:
                self.topology.clear()
            except Exception as error:
                errors.append(f"discovery cleanup: {error}")
        if self.monitoring:
            try:
                self.monitoring.write_collection_result(errors)
            except Exception as error:
                errors.append(f"collection metadata: {error}")
        return errors
