"""Client JVMs and the one current playback; Java client0 owns traffic coordination."""
import asyncio
from dataclasses import dataclass
import json
from typing import Annotated
from ..config.registry import Bounds, schema
from ..config.validation import validate
from .transport import ExecutionError, post


@dataclass(frozen=True)
class ClientResult:
    done: bool
    stopped: bool
    produced: Annotated[int, Bounds(minimum=0)]
    issued: Annotated[int, Bounds(minimum=0)]
    finished: Annotated[int, Bounds(minimum=0)]
    unsent: Annotated[int, Bounds(minimum=0)]
    queued: Annotated[int, Bounds(minimum=0)]
    inflight: Annotated[int, Bounds(minimum=0)]
    outcomes: dict[str, Annotated[int, Bounds(minimum=0)]]
    errors: list[str]


class ClientGroup:
    def __init__(self, *, component, recipe, launcher, supervisor, topology, records, monitoring):
        self.component, self.recipe, self.launcher = component, recipe, launcher
        self.supervisor, self.topology, self.records, self.monitoring = supervisor, topology, records, monitoring
        self.pending, self.stop_timeout = False, 30
        self._owned = []

    async def start_nodes(self):
        if self.component is None:
            return
        for index in range(self.component['deployment']['replicas']):
            name = f'client-{index}'
            self._owned.append(name)
            await self.launcher.start('client', index, self.recipe)
            if self.monitoring:
                self.monitoring.discover(name, self.records.directory / 'components' / name)

    def coordinator(self):
        nodes = [node for node in self.topology.active('client') if node['id'] == 'client-0']
        if len(nodes) != 1:
            raise ExecutionError('Expected exactly one client-0 coordinator')
        return nodes[0]['control']

    async def control(self, operation, args, query_time, timeout):
        envelope = {'interface': operation, 'args': args, 'context': {'run': 'foreground',
                    'query_time': query_time, 'observation': None, 'timeout_ms': max(1, int(timeout * 1000))}}
        if len(json.dumps(envelope).encode()) > 1024 * 1024:
            raise ExecutionError('Client control parameters exceed the 1 MiB limit')
        return await post(self.coordinator(), envelope)

    async def start(self, body, query_time, timeout):
        clients = self.topology.active('client')
        if not clients or body['concurrency'] < len(clients):
            raise ExecutionError('Need at least one concurrency slot per Client')
        available = {node['rpc'] for node in self.topology.active('master')}
        if not available or 'masters' in body and set(body['masters']) - available:
            raise ExecutionError('Unknown Master address')
        if self.pending:
            raise ExecutionError('Collect previous playback with client/stop before starting again')
        # A lost acknowledgement may still leave Java sending; teardown must attempt collection.
        self.pending = True
        self.stop_timeout = body['drain_timeout_ms'] / 1000 + 15 + len(clients) * 6
        result = await self.control('client/start', body, query_time, timeout)
        if result != {'accepted': True}:
            raise ExecutionError('Invalid Client start acknowledgement')
        return True

    async def collect(self, query_time, timeout):
        result = await self.control('client/stop', {}, query_time, timeout)
        self.records.emit(event='client_result', result=result)
        if not isinstance(result, dict) or set(result) != {'clients', 'errors'}:
            raise ExecutionError('Invalid Client result')
        validate(result['clients'], schema(dict[str, ClientResult]), 'client/stop.result')
        validate(result['errors'], schema(list[str]), 'client/stop.errors')
        if not result['clients'] and not result['errors'] and not self.pending:
            return {}
        clients = self.topology.active('client')
        if result['errors'] or set(result['clients']) != {node['control'] for node in clients} or any(
                not value['done'] or value['errors'] or value['inflight'] or value['queued']
                or value['issued'] != value['finished'] or sum(value['outcomes'].values()) != value['finished']
                or set(value['outcomes']) != {'success', 'error', 'timeout', 'cancelled'} for value in result['clients'].values()):
            raise ExecutionError(f'Client playback failed or results incomplete: {result}')
        self.pending = False
        by_address = {node['control']: node['rpc'] for node in clients}
        return {by_address[url]: value for url, value in result['clients'].items()}

    async def drain(self):
        if self.pending:
            from .records import timestamp
            async with asyncio.timeout(self.stop_timeout):
                await self.collect(timestamp(), self.stop_timeout)

    async def close(self):
        return await self.supervisor.stop(self._owned, 3)
