"""Explicit builtin adapter registration. Scenario data never imports Python."""

from .actions.admission import HANDLERS as ADMISSION_HANDLERS
from .actions.balance import HANDLERS as BALANCE_HANDLERS
from .actions.cancel import HANDLERS as CANCEL_HANDLERS
from .actions.elastic import HANDLERS as ELASTIC_HANDLERS
from .actions.engine_control import HANDLERS as ENGINE_CONTROL_HANDLERS
from .actions.engine_fault import HANDLERS as ENGINE_FAULT_HANDLERS
from .actions.engine_recovery import HANDLERS as ENGINE_RECOVERY_HANDLERS
from .actions.kv import HANDLERS as KV_HANDLERS
from .actions.kv_capacity import HANDLERS as KV_CAPACITY_HANDLERS
from .actions.kv_measurement import HANDLERS as KV_MEASUREMENT_HANDLERS
from .actions.master import HANDLERS as MASTER_HANDLERS
from .actions.observation import HANDLERS as OBSERVATION_HANDLERS
from .actions.priority import HANDLERS as PRIORITY_HANDLERS
from .actions.priority_preemption import HANDLERS as PRIORITY_PREEMPTION_HANDLERS
from .actions.rpc_measurement import HANDLERS as RPC_MEASUREMENT_HANDLERS
from .actions.status_protocol import HANDLERS as STATUS_PROTOCOL_HANDLERS


def handlers():
    result = {}
    for descriptor in [
        *ADMISSION_HANDLERS,
        *BALANCE_HANDLERS,
        *CANCEL_HANDLERS,
        *ELASTIC_HANDLERS,
        *ENGINE_CONTROL_HANDLERS,
        *ENGINE_FAULT_HANDLERS,
        *ENGINE_RECOVERY_HANDLERS,
        *KV_HANDLERS,
        *KV_CAPACITY_HANDLERS,
        *KV_MEASUREMENT_HANDLERS,
        *MASTER_HANDLERS,
        *OBSERVATION_HANDLERS,
        *PRIORITY_HANDLERS,
        *PRIORITY_PREEMPTION_HANDLERS,
        *RPC_MEASUREMENT_HANDLERS,
        *STATUS_PROTOCOL_HANDLERS,
    ]:
        if descriptor.name in result:
            raise ValueError(f"duplicate action {descriptor.name}")
        result[descriptor.name] = descriptor
    return result
