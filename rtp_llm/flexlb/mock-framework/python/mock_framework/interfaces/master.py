"""Case-level Master reads; Java returns only its own observation."""
from dataclasses import dataclass
from typing import Annotated, Literal
import asyncio
import json
from urllib.request import Request, urlopen
from ..config.registry import Bounds, case_interface
from ..execution.transport import post, ExecutionError


@dataclass(frozen=True)
class Targets:
    targets: list[str] | None = None

    def __post_init__(self):
        if self.targets is not None and (not self.targets or len(set(self.targets)) != len(self.targets)):
            raise ValueError('targets must be nonempty and unique when supplied')


@dataclass(frozen=True)
class EngineSelection(Targets):
    role: Literal['prefill', 'decode', 'vit'] = 'prefill'


def selected(ctx, targets):
    available = {item['rpc']: item for item in ctx.runtime.topology.active('master')}
    addresses = sorted(available) if targets is None else targets
    if not addresses or set(addresses) - available.keys():
        raise ExecutionError('Master targets are empty or unknown')
    return [available[address] for address in addresses]


async def read(ctx, master, operation, args):
    return await post(master['control'], {'interface': operation, 'args': args,
        'context': {'run': 'foreground', 'query_time': ctx.query_time, 'observation': None,
                    'timeout_ms': max(1, int(ctx.timeout * 1000))}})


@case_interface('master/leadership/get', effect='read')
async def leadership(ctx, args: Targets) -> list[str]:
    """Return each selected Master's actual leadership claim; do not invent a unique leader."""
    result = []
    for master in selected(ctx, args.targets):
        value = await read(ctx, master, 'master/leadership/get', {})
        own = value['instances']
        if own not in ([], [{'id': master['id']}]):
            raise ExecutionError('Master returned an invalid local leadership observation')
        if own:
            result.append(master['rpc'])
    return result


@case_interface('master/engine/get', effect='read')
async def engines(ctx, args: EngineSelection) -> dict[str, list[str]]:
    """Keep each Master's discovered Engine addresses separate."""
    result = {}
    for master in selected(ctx, args.targets):
        value = await read(ctx, master, 'master/engine/get', {'role': args.role})
        if set(value['instances']) != {master['id']}:
            raise ExecutionError('Master returned another instance observation')
        addresses = value['instances'][master['id']]
        if not isinstance(addresses, list) or any(not isinstance(address, str) or not address for address in addresses):
            raise ExecutionError('Master reported invalid Engine addresses')
        if len(set(addresses)) != len(addresses):
            raise ExecutionError('Master reported duplicate Engines')
        result[master['rpc']] = sorted(addresses)
    return result


@dataclass(frozen=True)
class Start:
    count: Annotated[int, Bounds(1, 1024)] = 1


@case_interface('master/start', effect='write')
async def start(ctx, args: Start) -> list[str]:
    """Create new Master JVMs in this run's election group; never reuse stopped identities."""
    return await ctx.runtime.masters.start(args.count)


@dataclass(frozen=True)
class Stop(Targets):
    mode: Literal['terminate', 'kill'] = 'terminate'


@case_interface('master/stop', effect='write')
async def stop(ctx, args: Stop) -> list[str]:
    """Stop selected Master JVMs, or all active Masters when targets is omitted."""
    return await ctx.runtime.masters.stop(args.targets, args.mode)


@case_interface('master/restart', effect='write')
async def restart(ctx, args: Targets) -> list[str]:
    """Restore stopped Masters at their original service addresses with fresh process artifacts."""
    return await ctx.runtime.masters.restart(args.targets)


def inflight(url, timeout):
    request = Request(url + '/rtp_llm/inflight_status', headers={'Accept': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024:
        raise ExecutionError('Master inflight response exceeds 16 MiB')
    try:
        data = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        values = [data['scheduler_inflight']]
        for side, fields in [('prefill', ('inflight_requests',)), ('decode', ('master_queued', 'confirmed_running'))]:
            endpoints = data[side + '_endpoints']
            if not isinstance(endpoints, list):
                raise ExecutionError('Master endpoint ledger must be a list')
            values.extend(endpoint[field] for endpoint in endpoints for field in fields)
        if any(type(value) is not int or value < 0 for value in values):
            raise ExecutionError('Master owner ledger requires nonnegative integer counts')
    except (KeyError, TypeError, ValueError) as error:
        raise ExecutionError('Invalid Master owner ledger') from error
    return all(value == 0 for value in values)


@case_interface('master/idle', effect='read')
async def idle(ctx, args: Targets) -> bool:
    """Read every selected Master's native owner ledger once; missing data is never zero."""
    results = []
    for master in selected(ctx, args.targets):
        results.append(await asyncio.to_thread(inflight, master['http'], min(2, ctx.timeout)))
    return all(results)
