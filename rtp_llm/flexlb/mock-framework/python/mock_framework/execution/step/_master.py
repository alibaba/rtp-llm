"""Case-level Master reads; Java returns only its own observation."""
from dataclasses import dataclass
from .. import ExecutionError
from ..transport import post





@dataclass(frozen=True)
class Targets:
    targets: list[str] | None = None

    def __post_init__(self):
        if self.targets is not None and (not self.targets or len(set(self.targets)) != len(self.targets)):
            raise ValueError('targets must be nonempty and unique when supplied')


def _select_masters(ctx, targets):
    available = {item['rpc']: item for item in ctx.runtime.topology.active('master')}
    addresses = sorted(available) if targets is None else targets
    if not addresses or set(addresses) - available.keys():
        raise ExecutionError('Master targets are empty or unknown')
    return [available[address] for address in addresses]


async def _read_master_state(ctx, master, operation, args):
    return await post(master['control'], {'interface': operation, 'args': args,
        'context': {'run': 'foreground', 'query_time': ctx.query_time, 'observation': None,
                    'timeout_ms': max(1, int(ctx.timeout * 1000))}})
