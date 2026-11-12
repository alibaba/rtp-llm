"""Physical Engine placement; Case operations address instances, never JVMs."""

import asyncio
import socket
from itertools import zip_longest

from ...config.validation import duration
from .. import ExecutionError
from ..processes import ProcessIdentity, start_instance
from . import register_ready_instance

ROLES = ("prefill", "decode", "vit")
INITIAL_START_CONCURRENCY = 32
MAX_ENGINE_START_COUNT = 1024


class EngineRole:
    def __init__(
        self, *, supervisor, topology, records, environment, launch_by_role, monitoring
    ):
        self.supervisor, self.topology, self.records = supervisor, topology, records
        self.role_configs, self.launch_by_role, self.monitoring = (
            environment,
            launch_by_role,
            monitoring,
        )
        self.counters = {}
        self.used_ports = set()
        self.lock = asyncio.Lock()

    def _port(self):
        # Binding is a probe, not a reservation; a subsequent bind failure fails the run.
        for _ in range(32):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
        raise ExecutionError("Cannot allocate a fresh Engine listener port")

    async def _start_instance(self, role, index):
        identity = ProcessIdentity(role, index)
        rpc_port, control_port = self._port(), self._port()
        child, control, endpoint = await start_instance(
            self.supervisor,
            self.records,
            self.role_configs[role],
            identity,
            self.launch_by_role[role],
            configure=lambda _: {
                "MOCK_BIND_HOST": "127.0.0.1",
                "MOCK_ADVERTISE_HOST": "127.0.0.1",
                "START_PORT": str(control_port),
                "MOCK_RPC_PORT": str(rpc_port),
            },
        )
        expected = {
            "id": identity.instance_id,
            "role": role,
            "rpc": f"127.0.0.1:{rpc_port}",
            "control": f"http://127.0.0.1:{control_port}",
        }
        if (
            endpoint is None
            or control != expected["control"]
            or any(endpoint.get(key) != value for key, value in expected.items())
        ):
            raise ExecutionError(
                "Engine endpoints differ from the assigned identity and listeners"
            )
        return identity, child.pid, control, endpoint

    def _register_instance(self, identity, pid, control, endpoint):
        register_ready_instance(
            self.topology,
            self.monitoring,
            self.records,
            identity,
            control,
            endpoint,
            pid=pid,
        )
        return endpoint["rpc"]

    async def scale_out(self, role, count):
        return await self._start_instances({role: count}, concurrency=1)

    async def scale_in(self, role, targets, count):
        async with self.lock:
            if role not in ROLES:
                raise ExecutionError("scale_in requires an Engine role")
            candidates = self.topology.active(role)
            known = {item["rpc"]: item for item in candidates}
            if targets is not None:
                if (
                    not targets
                    or len(set(targets)) != len(targets)
                    or set(targets) - known.keys()
                ):
                    raise ExecutionError(
                        "Engine targets must be nonempty, unique, active addresses of the selected role"
                    )
                candidates = [known[address] for address in targets]
            if (
                not candidates
                or count is not None
                and not 1 <= count <= len(candidates)
            ):
                raise ExecutionError(
                    "No Engine candidates or scale_in count exceeds the selected set"
                )
            selected = candidates if count is None else candidates[:count]
            # Selection is fixed once, before stopping any process.
            self.records.emit(
                event="instances_selected",
                operation="scale_in",
                targets=[x["rpc"] for x in selected],
            )
            if self.monitoring:
                await self.monitoring.wait_for_final_scrape(
                    [item["id"] for item in selected]
                )
            for item in selected:
                node = item["id"]
                child = self.supervisor.child(node)
                errors = await self.supervisor.stop_intentionally(node)
                if child.returncode is not None:
                    ready = self.records.directory / "components" / node / "ready.json"
                    ready.unlink(missing_ok=True)
                    self.topology.retire(node)
                    if self.monitoring:
                        self.monitoring.retire(node)
                    self.records.emit(
                        event="instance_stopped",
                        node=node,
                        rpc=item["rpc"],
                        role=role,
                        instance_id=node,
                        generation=0,
                    )
                if errors:
                    raise ExecutionError("; ".join(errors))
            return [item["rpc"] for item in selected]

    async def start(self):
        counts = {
            role: self.role_configs[role]["deployment"]["replicas"]
            for role in self.launch_by_role
            if role in ROLES
        }
        if counts:
            timeout = sum(
                float(duration(self.launch_by_role[role]["startup_timeout"])) * count
                for role, count in counts.items()
            )
            await self.supervisor.guard(
                self._start_instances(counts, INITIAL_START_CONCURRENCY), timeout
            )

    async def _start_instances(self, counts, concurrency):
        async with self.lock:
            if any(
                role not in ROLES
                or role not in self.role_configs
                or not 1 <= count <= MAX_ENGINE_START_COUNT
                for role, count in counts.items()
            ):
                raise ExecutionError(
                    f"Engine startup requires configured roles and counts in 1..{MAX_ENGINE_START_COUNT}"
                )
            next_registration = {role: self.counters.get(role, 0) for role in counts}
            jobs_by_role = [
                [(role, index) for index in range(first, first + counts[role])]
                for role, first in next_registration.items()
            ]

            # Interleave roles while keeping a single budget for all in-flight starts.
            jobs = (
                job
                for group in zip_longest(*jobs_by_role)
                for job in group
                if job is not None
            )
            ready, result = {}, []

            async def worker():
                for role, index in jobs:
                    self.counters[role] = index + 1
                    ready[role, index] = await self._start_instance(role, index)
                    # Keep scale-in selection stable even if readiness arrives out of order.
                    first = next_registration[role]
                    while (role, first) in ready:
                        result.append(
                            self._register_instance(*ready.pop((role, first)))
                        )
                        first += 1
                    next_registration[role] = first

            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(min(concurrency, sum(counts.values()))):
                        group.create_task(worker())
            except ExceptionGroup as errors:
                raise errors.exceptions[0] from errors
            if self.monitoring and self.monitoring.baseline_ready:
                await self.monitoring.wait_for_baseline(
                    [
                        item["id"]
                        for item in self.topology.active()
                        if item["rpc"] in result
                    ]
                )
            return result

    async def close(self):
        processes = self.supervisor.process_identities(*ROLES)
        # All Engine JVMs shut down concurrently, including in full-scale Cases.
        errors = await self.supervisor.stop(
            [process.process_name for process in processes], 15
        )
        remaining = set(self.supervisor.process_identities(*ROLES))
        for process in processes:
            if process not in remaining:
                (process.directory(self.records.directory) / "ready.json").unlink(
                    missing_ok=True
                )
        return errors
