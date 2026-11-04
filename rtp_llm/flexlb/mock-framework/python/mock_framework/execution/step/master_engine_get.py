from dataclasses import dataclass
from typing import Literal
from .registry import register_step
from .step import Step, StepContext
from .. import ExecutionError
from ._master import Targets, _read_master_state, _select_masters


@dataclass(frozen=True)
class EngineSelection(Targets):
    role: Literal['prefill', 'decode', 'vit'] = 'prefill'


@register_step('master/engine/get', effect='read')
class MasterEngineGet(Step):
    @staticmethod
    async def execute(ctx: StepContext, args: EngineSelection) -> dict[str, list[str]]:
        """Keep each Master's discovered Engine addresses separate."""
        result = {}
        for master in _select_masters(ctx, args.targets):
            value = await _read_master_state(ctx, master, 'master/engine/get', {'role': args.role})
            if set(value['instances']) != {master['id']}:
                raise ExecutionError('Master returned another instance observation')
            addresses = value['instances'][master['id']]
            if not isinstance(addresses, list) or any(not isinstance(address, str) or not address for address in addresses):
                raise ExecutionError('Master reported invalid Engine addresses')
            if len(set(addresses)) != len(addresses):
                raise ExecutionError('Master reported duplicate Engines')
            result[master['rpc']] = sorted(addresses)
        return result
