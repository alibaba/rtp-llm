"""Native Master configuration binding and per-Master state reads."""

import asyncio
import errno
import json
import os
import secrets
import socket
import uuid
from dataclasses import replace
from urllib.parse import urlsplit

from .. import ExecutionError
from ..processes import ProcessIdentity, start_instance
from ..step_trace import write_json
from ..transport import invoke
from . import ENGINE_ROLES, register_ready_instance, retire_instance


def _resolve_discovery(route_json, topology, *, require_engines):
    if not isinstance(route_json, str):
        raise ExecutionError(
            "Master native setting MODEL_SERVICE_CONFIG must be an environment string"
        )
    try:
        route = json.loads(route_json)
        groups = route["role_endpoints"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise ValueError(
                "local Master currently requires exactly one endpoint group"
            )
        hosts = {}
        configured = {node["role"] for node in topology}
        for role in ENGINE_ROLES:
            endpoint = groups[0].get(role + "_endpoint")
            if endpoint is None:
                if role in configured:
                    raise ValueError(f"Missing native {role}_endpoint")
                continue
            if (
                endpoint.get("protocol") != "grpc"
                or not isinstance(endpoint.get("address"), str)
                or not endpoint["address"]
            ):
                raise ValueError(
                    "local Engine endpoints require grpc protocol and a discovery address"
                )
            domain = endpoint["address"]
            if domain in hosts:
                raise ValueError("Each Engine role needs a distinct discovery address")
            addresses = []
            for node in topology:
                if node["role"] == role:
                    host, port = node["rpc"].rsplit(":", 1)
                    # Native WorkerAddressService translates grpc discovery ports to its HTTP identity.
                    port = int(port)
                    if port <= 0:
                        raise ValueError("Invalid Engine RPC port")
                    addresses.append(f"{host}:{port}")
            if require_engines and not addresses:
                raise ValueError(f"No running Engines for native {role}_endpoint")
            hosts[domain] = addresses
        if not hosts or groups[0].get("pd_fusion_endpoint") is not None:
            raise ValueError("local Master supports separate P/D/V endpoints only")
        return route, hosts
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ExecutionError(
            f"Cannot bind local MODEL_SERVICE_CONFIG: {error}"
        ) from error


def prepare_master_env(config, topology, directory, *, require_engines=True):
    env = {}
    for key, value in config.items():
        if key.isupper():
            if not isinstance(value, str):
                raise ExecutionError(
                    f"Master native setting {key} must be an environment string"
                )
            env[key] = value
    if "MODEL_SERVICE_CONFIG" not in env:
        return env  # Generic transport probes do not launch the native Master.
    route, hosts = _resolve_discovery(
        env["MODEL_SERVICE_CONFIG"], topology, require_engines=require_engines
    )
    discovery = directory / "discovery.json"
    write_json(discovery, hosts)
    route.pop("hosts", None)
    route["discovery_file"] = str(discovery)
    env["MODEL_SERVICE_CONFIG"] = json.dumps(route, ensure_ascii=False)
    return env


def refresh_discovery(config, topology, directory):
    if "MODEL_SERVICE_CONFIG" in config:
        _, hosts = _resolve_discovery(
            config["MODEL_SERVICE_CONFIG"], topology, require_engines=False
        )
        write_json(directory / "discovery.json", hosts)


class MasterRole:
    def __init__(
        self,
        *,
        role_config,
        launch_config,
        election,
        supervisor,
        topology,
        records,
        monitoring,
    ):
        self.role_config, self.launch_config, self.election = (
            role_config,
            launch_config,
            election,
        )
        self.supervisor, self.topology = supervisor, topology
        self.records, self.monitoring = records, monitoring
        self.election_env = {}
        self._next = 0
        self._instances = {}
        self._lock = asyncio.Lock()

    def configure_election(self, connect):
        self.election_env = {
            "HIPPO_ROLE": "mock-" + uuid.uuid4().hex,
            "FLEXLB_SYNC_CONSISTENCY_CONFIG": json.dumps(
                {
                    "needConsistency": True,
                    "zookeeperConfig": {"zkHost": connect, "zkTimeoutMs": 6000},
                }
            ),
            "MOCK_MASTER_PORT": str(
                master_port(self.role_config["deployment"]["replicas"])
            ),
        }

    def _prepare_env(self, index, directory, engines):
        env = prepare_master_env(
            self.role_config["config"],
            engines,
            directory,
            require_engines=index < self.role_config["deployment"]["replicas"],
        )
        if self.election_env:
            host = master_host(index)
            check_master_ports(host, int(self.election_env["MOCK_MASTER_PORT"]))
            env.update(self.election_env, FLEXLB_ADVERTISED_IP=host)
        return env

    def _validate_endpoint(self, instance_id, index, endpoint, control):
        if not self.election_env:
            return
        host, port = master_host(index), int(self.election_env["MOCK_MASTER_PORT"])
        address = urlsplit(control)
        if (
            address.scheme != "http"
            or address.hostname != host
            or not address.port
            or address.port in (port, port + 2)
        ):
            raise ExecutionError("Master control address differs from assigned host")
        expected = {
            "id": instance_id,
            "role": "master",
            "http": f"http://{host}:{port}",
            "rpc": f"{host}:{port + 2}",
            "control": control,
            "metrics": f"http://{host}:{port}/prometheus",
        }
        if endpoint != expected:
            raise ExecutionError(
                "Master endpoints differ from assigned identity and listeners"
            )

    def publish_discovery(self, snapshot):
        if self.role_config is None:
            return
        processes = {
            process.instance_id: process
            for process in self.supervisor.process_identities("master")
        }
        for node in snapshot:
            if node["role"] == "master":
                directory = processes[node["id"]].directory(self.records.directory)
                refresh_discovery(self.role_config["config"], snapshot, directory)

    async def _start_instance(self, identity, *, expected_address=None):
        def engine_endpoints():
            return [
                node
                for node in self.topology.list_active_endpoints()
                if node["role"] in ENGINE_ROLES
            ]

        engines_at_start = engine_endpoints()
        _, control, endpoint = await start_instance(
            self.supervisor,
            self.records,
            self.role_config,
            identity,
            self.launch_config,
            configure=lambda directory: self._prepare_env(
                identity.index, directory, engines_at_start
            ),
        )
        if endpoint is None and self.election_env:
            raise ExecutionError("Native Master did not publish endpoints")
        if expected_address is not None and (
            endpoint is None or endpoint["rpc"] != expected_address
        ):
            raise ExecutionError("Restarted Master changed its service address")
        if endpoint is not None:
            self._validate_endpoint(
                identity.instance_id, identity.index, endpoint, control
            )
            # An Engine change during startup cannot see this Master in the registry.
            engines = engine_endpoints()
            if engines != engines_at_start:
                refresh_discovery(
                    self.role_config["config"],
                    engines,
                    identity.directory(self.records.directory),
                )
        register_ready_instance(
            self.topology, self.monitoring, self.records, identity, control, endpoint
        )
        if self.monitoring and self.monitoring.baseline_ready:
            await self.monitoring.wait_for_baseline([identity.instance_id])
        if endpoint is not None:
            self._instances[endpoint["rpc"]] = identity
        return endpoint

    async def start(self, count=None):
        if count is None:
            if self.role_config is None:
                return []
            count = self.role_config["deployment"]["replicas"]
        async with self._lock:
            if self.launch_config is None or not 1 <= count <= 1024:
                raise ExecutionError(
                    "master/start requires configured Masters and count in 1..1024"
                )
            result = []
            for _ in range(count):
                index = self._next
                self._next += 1
                endpoint = await self._start_instance(ProcessIdentity("master", index))
                if endpoint is not None:
                    result.append(endpoint["rpc"])
            return result

    def _select_endpoints(self, targets):
        available = {
            item["rpc"]: item for item in self.topology.list_active_endpoints("master")
        }
        addresses = sorted(available) if targets is None else targets
        if not addresses or set(addresses) - available.keys():
            raise ExecutionError("Master targets are empty or unknown")
        return [available[address] for address in addresses]

    async def list_addresses(self, targets, is_leader, *, remaining):
        masters = self._select_endpoints(targets)
        if is_leader is None:
            return [master["rpc"] for master in masters]
        result = []
        for master in masters:
            value = await invoke(
                master["control"], "master/leadership/get", {}, remaining()
            )
            own = value["instances"]
            if own not in ([], [{"id": master["id"]}]):
                raise ExecutionError(
                    "Master returned an invalid local leadership observation"
                )
            if bool(own) == is_leader:
                result.append(master["rpc"])
        return result

    async def list_engine_addresses(self, targets, role, *, remaining):
        result = {}
        for master in self._select_endpoints(targets):
            value = await invoke(
                master["control"], "master/engine/get", {"role": role}, remaining()
            )
            if set(value["instances"]) != {master["id"]}:
                raise ExecutionError("Master returned another instance observation")
            addresses = value["instances"][master["id"]]
            if not isinstance(addresses, list) or any(
                not isinstance(address, str) or not address for address in addresses
            ):
                raise ExecutionError("Master reported invalid Engine addresses")
            if len(set(addresses)) != len(addresses):
                raise ExecutionError("Master reported duplicate Engines")
            result[master["rpc"]] = sorted(addresses)
        return result

    def _select_addresses(self, targets, *, active):
        available = {
            node["rpc"] for node in self.topology.list_active_endpoints("master")
        }
        candidates = available if active else self._instances.keys() - available
        if (
            not targets
            or len(set(targets)) != len(targets)
            or set(targets) - candidates
        ):
            state = "active" if active else "stopped"
            raise ExecutionError(
                f"Master targets must be nonempty, unique, {state} addresses"
            )
        return list(targets)

    async def stop(self, targets, mode="terminate"):
        async with self._lock:
            if mode not in ("terminate", "kill"):
                raise ExecutionError("Unknown Master stop mode")
            addresses = self._select_addresses(targets, active=True)
            selected = [self._instances[address] for address in addresses]
            if any(
                self.supervisor.child(item.process_name).returncode is not None
                for item in selected
            ):
                raise ExecutionError("Selected Master is not running")
            self.records.emit(
                event="instances_selected",
                operation="master/stop",
                targets=addresses,
                mode=mode,
            )
            if self.monitoring:
                await self.monitoring.wait_for_final_scrape(
                    [item.instance_id for item in selected]
                )
            for item in selected:
                child = self.supervisor.child(item.process_name)
                errors = await self.supervisor.stop_intentionally(
                    item.process_name, mode=mode, grace=self._stop_grace()
                )
                if child.returncode is not None:
                    retire_instance(
                        self.topology,
                        self.monitoring,
                        item.instance_id,
                        item.directory(self.records.directory),
                    )
                    self.records.emit(
                        event="instance_stopped",
                        node=item.instance_id,
                        process=item.process_name,
                        **item.event_fields(),
                    )
                if errors:
                    raise ExecutionError("; ".join(errors))
            return addresses

    async def restart(self, targets):
        async with self._lock:
            addresses = self._select_addresses(targets, active=False)
            for address in addresses:
                item = self._instances[address]
                await self._start_instance(
                    replace(item, generation=item.generation + 1),
                    expected_address=address,
                )
            return addresses

    def _stop_grace(self):
        # Native shutdown includes up to 30s of leadership transfer, quiet time and resource teardown.
        return 60 if self.election is not None else 3

    async def close(self):
        errors = []
        # Keep peers and ZooKeeper alive until each Master has completed its shutdown.
        for process in reversed(self.supervisor.process_identities("master")):
            errors.extend(
                await self.supervisor.stop([process.process_name], self._stop_grace())
            )
        return errors


def master_port(count):
    """All Masters use distinct loopback IPv4 addresses and the same native ports."""
    if not 1 <= count <= 254 * 256:
        raise ExecutionError(
            "Too many local Masters for the loopback address allocation"
        )
    for attempt in range(32):
        sockets = []
        # Avoid client connections taking the selected port during JVM startup.
        port = 10000 + secrets.randbelow(20000)
        try:
            for index in range(count):
                host = f"127.0.{index // 254}.{index % 254 + 1}"
                for offset in (0, 2):
                    sock = socket.socket()
                    sockets.append(sock)
                    sock.bind((host, port + offset))
            return port
        except OSError as error:
            if error.errno == errno.EADDRNOTAVAIL:
                raise ExecutionError(
                    f"Cannot bind {host}: enable loopback aliases or run on Linux"
                ) from error
            if attempt == 31:
                raise ExecutionError(
                    "No available shared Master HTTP/gRPC port pair"
                ) from error
        finally:
            for sock in sockets:
                sock.close()


def master_host(index):
    if not 0 <= index < 254 * 256:
        raise ExecutionError("Local Master identity range exhausted")
    return f"127.0.{index // 254}.{index % 254 + 1}"


def check_master_ports(host, port):
    try:
        for offset in (0, 2):
            with socket.socket() as probe:
                # Match the native listener: TIME_WAIT must not prevent an owned address from restarting.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, port + offset))
    except OSError as error:
        detail = (
            "enable loopback aliases or run on Linux"
            if error.errno == errno.EADDRNOTAVAIL
            else str(error)
        )
        raise ExecutionError(f"Cannot bind Master {host}:{port}: {detail}") from error
