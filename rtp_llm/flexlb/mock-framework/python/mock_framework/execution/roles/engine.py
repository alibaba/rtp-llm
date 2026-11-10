"""Physical Engine placement; Case operations address instances, never JVMs."""
import asyncio
import socket
from itertools import zip_longest
from ...config.validation import duration
from ..processes import launch_component
from .. import ExecutionError

ROLES = ('prefill', 'decode', 'vit')
INITIAL_START_CONCURRENCY = 32


class EngineRole:
    def __init__(self, *, supervisor, topology, records, environment, recipes, monitoring):
        self.supervisor, self.topology, self.records = supervisor, topology, records
        self.environment, self.recipes, self.monitoring = environment, recipes, monitoring
        self.counters = {}
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

    async def _start_instance(self, role, index):
        node = f'{role}-{index}'
        rpc_port, control_port = self._port(), self._port()
        self.processes.add(node)
        child, control, endpoint = await launch_component(
            self.supervisor, self.records, self.environment, role, index, self.recipes[role],
            configure=lambda _: {
                'MOCK_BIND_HOST': '127.0.0.1', 'MOCK_ADVERTISE_HOST': '127.0.0.1',
                'START_PORT': str(control_port), 'MOCK_RPC_PORT': str(rpc_port),
            })
        expected = {'id': node, 'role': role, 'rpc': f'127.0.0.1:{rpc_port}', 'control': f'http://127.0.0.1:{control_port}'}
        if endpoint is None or control != expected['control'] or any(endpoint.get(key) != value for key, value in expected.items()):
            raise ExecutionError('Engine endpoints differ from the assigned identity and listeners')
        return child.pid, control, endpoint

    def _register_instance(self, pid, control, endpoint):
        node, role, rpc = endpoint['id'], endpoint['role'], endpoint['rpc']
        self.topology.register(node, role, control, endpoint)
        self.records.emit(event='instance_started', node=node, pid=pid, rpc=rpc,
                          role=role, instance_id=node, generation=0)
        if self.monitoring:
            self.monitoring.discover(node, self.records.directory / 'components' / node)
        return rpc

    async def scale_out(self, role, count):
        return await self._start_roles({role: count}, concurrency=1)

    async def scale_in(self, role, targets, count):
        async with self.lock:
            if role not in ROLES:
                raise ExecutionError('scale_in requires an Engine role')
            candidates = self.topology.active(role)
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
                self.topology.retire(node)
                if self.monitoring:
                    self.monitoring.retire(node)
                self.records.emit(event='instance_stopped', node=node, rpc=item['rpc'],
                                  role=role, instance_id=node, generation=0)
            return [item['rpc'] for item in selected]

    async def start(self):
        counts = {role: self.environment[role]['deployment']['replicas'] for role in self.recipes if role in ROLES}
        if counts:
            timeout = sum(float(duration(self.recipes[role]['startup_timeout'])) * count for role, count in counts.items())
            await self.supervisor.guard(self._start_roles(counts, INITIAL_START_CONCURRENCY), timeout)

    async def _start_roles(self, counts, concurrency):
        async with self.lock:
            if any(role not in ROLES or role not in self.environment or not 1 <= count <= 1024
                   for role, count in counts.items()):
                raise ExecutionError('Engine startup requires configured roles and counts in 1..1024')
            next_registration = {role: self.counters.get(role, 0) for role in counts}
            jobs_by_role = [[(role, index) for index in range(first, first + counts[role])]
                            for role, first in next_registration.items()]

            # Interleave roles while keeping a single budget for all in-flight starts.
            jobs = (job for group in zip_longest(*jobs_by_role) for job in group if job is not None)
            ready, result = {}, []

            async def worker():
                for role, index in jobs:
                    self.counters[role] = index + 1
                    ready[role, index] = await self._start_instance(role, index)
                    # Keep scale-in selection stable even if readiness arrives out of order.
                    first = next_registration[role]
                    while (role, first) in ready:
                        result.append(self._register_instance(*ready.pop((role, first))))
                        first += 1
                    next_registration[role] = first

            try:
                async with asyncio.TaskGroup() as group:
                    for _ in range(min(concurrency, sum(counts.values()))):
                        group.create_task(worker())
            except ExceptionGroup as errors:
                raise errors.exceptions[0] from errors
            if self.monitoring and self.monitoring.baselined:
                await self.monitoring.baseline([item['id'] for item in self.topology.active() if item['rpc'] in result])
            return result

    async def close(self):
        names = [name for name, _ in self.supervisor.children if name in self.processes]
        # All Engine JVMs shut down concurrently, including in full-scale Cases.
        errors = await self.supervisor.stop(names, 15)
        active = {name for name, _ in self.supervisor.children}
        for node in self.processes - active:
            (self.records.directory / 'components' / node / 'ready.json').unlink(missing_ok=True)
        return errors
