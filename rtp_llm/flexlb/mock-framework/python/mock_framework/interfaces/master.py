"""Case-level Master reads; Java returns only its own observation."""
from dataclasses import dataclass
from typing import Literal
from ..config.registry import case_interface
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
    available = {item['rpc']: item for item in ctx.processes.topology if item['role'] == 'master'}
    addresses = sorted(available) if targets is None else targets
    if not addresses or set(addresses) - available.keys():
        raise ExecutionError('Master targets are empty or unknown')
    return [available[address] for address in addresses]


async def read(ctx, master, operation, args):
    return await post(master['control'], {'interface': operation, 'args': args,
        'context': {'run': 'foreground', 'query_time': ctx.query_time, 'observation': None,
                    'timeout_ms': max(1, int(ctx.timeout * 1000))}})


def identities(ctx):
    return {item['id']: item['rpc'] for item in ctx.processes.history}


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
    known, result = identities(ctx), {}
    for master in selected(ctx, args.targets):
        value = await read(ctx, master, 'master/engine/get', {'role': args.role})
        if set(value['instances']) != {master['id']}:
            raise ExecutionError('Master returned another instance observation')
        try:
            addresses = [known[item['id']] for item in value['instances'][master['id']]['instances']]
        except KeyError as error:
            raise ExecutionError('Master reported an unknown Engine') from error
        if len(set(addresses)) != len(addresses):
            raise ExecutionError('Master reported duplicate Engines')
        result[master['rpc']] = sorted(addresses)
    return result
