from dataclasses import dataclass
from .registry import register_step
from .step import Step, StepContext
from .. import ExecutionError
from ._master import Targets, _read_master_state, _select_masters


@dataclass(frozen=True)
class MasterSelection(Targets):
    is_leader: bool | None = None


@register_step('master/get', effect='read')
class MasterGet(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: MasterSelection) -> list[str]:
        """List active Masters, optionally filtering their actual leadership claims."""
        masters = _select_masters(ctx, args.targets)
        if args.is_leader is None:
            return [master['rpc'] for master in masters]
        result = []
        for master in masters:
            value = await _read_master_state(ctx, master, 'master/leadership/get', {})
            own = value['instances']
            if own not in ([], [{'id': master['id']}]):
                raise ExecutionError('Master returned an invalid local leadership observation')
            if bool(own) == args.is_leader:
                result.append(master['rpc'])
        return result
